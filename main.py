import os
import time

from absl import logging

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from environment import GameEnvironment
import tensorflow as tf

# Validate environment
# env = GameEnvironment(3)
# utils.validate_py_environment(env, episodes=5)
from network import SevenWondersActorNetwork, SevenWondersValueNetwork

tf.compat.v1.enable_v2_behavior()

number_of_players = 3
actor_fc_layers = (200, 100)
value_fc_layers = (200, 100)
# Params for collect
num_environment_steps = 10000000
collect_episodes_per_iteration = 30
num_parallel_environments = 1
replay_buffer_capacity = 1001  # Per-environment
# Params for train
num_epochs = 25
learning_rate = 1e-4
# Params for eval
num_eval_episodes = 30
eval_interval = 500
# Params for summaries and logging
log_interval = 50
summary_interval = 50
summaries_flush_secs = 1
use_tf_functions = True
debug_summaries = False
summarize_grads_and_vars = False

root_dir = "logs"
train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'eval')

# Start training
train_summary_writer = tf.compat.v2.summary.create_file_writer(
    train_dir, flush_millis=summaries_flush_secs * 1000)
train_summary_writer.set_as_default()

eval_summary_writer = tf.compat.v2.summary.create_file_writer(
    eval_dir, flush_millis=summaries_flush_secs * 1000)
eval_metrics = [
    tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
    tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
]

global_step = tf.compat.v1.train.get_or_create_global_step()
with tf.compat.v2.summary.record_if(
        lambda: tf.math.equal(global_step % summary_interval, 0)):
    tf_env = tf_py_environment.TFPyEnvironment(GameEnvironment(number_of_players))
    eval_tf_env = tf_py_environment.TFPyEnvironment(GameEnvironment(number_of_players))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    actor_net = SevenWondersActorNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=actor_fc_layers)

    value_net = SevenWondersValueNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=value_fc_layers)

    tf_agent = ppo_agent.PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=num_epochs,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()

    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_buffer_capacity)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration)


    def train_step():
        trajectories = replay_buffer.gather_all()
        return tf_agent.train(experience=trajectories)


    if use_tf_functions:
        # TODO(b/123828980): Enable once the cause for slowdown was identified.
        collect_driver.run = common.function(collect_driver.run, autograph=False)
        tf_agent.train = common.function(tf_agent.train, autograph=False)
        train_step = common.function(train_step)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    while environment_steps_metric.result() < num_environment_steps:
        global_step_val = global_step.numpy()
        if global_step_val % eval_interval == 0:
            metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics',
            )

        start_time = time.time()
        collect_driver.run()
        collect_time += time.time() - start_time

        start_time = time.time()
        total_loss, _ = train_step()
        replay_buffer.clear()
        train_time += time.time() - start_time

        for train_metric in train_metrics:
            train_metric.tf_summaries(
                train_step=global_step, step_metrics=step_metrics)

        if global_step_val % log_interval == 0:
            logging.info('step = %d, loss = %f', global_step_val, total_loss)
            steps_per_sec = (
                    (global_step_val - timed_at_step) / (collect_time + train_time))
            logging.info('%.3f steps/sec', steps_per_sec)
            logging.info('collect_time = {}, train_time = {}'.format(
                collect_time, train_time))
            with tf.compat.v2.summary.record_if(True):
                tf.compat.v2.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_sec, step=global_step)

            timed_at_step = global_step_val
            collect_time = 0
            train_time = 0

    # One final eval before exiting.
    metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )
