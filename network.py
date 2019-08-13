from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import encoding_network, categorical_projection_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils


class SevenWondersActorNetwork(network.DistributionNetwork):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 name='SevenWondersActorNetwork'):

        projection_networks = []
        for single_output_spec in tf.nest.flatten(action_spec):
            projection_networks.append(categorical_projection_network.CategoricalProjectionNetwork(
                single_output_spec, logits_init_output_factor=0.1))

        projection_distribution_specs = [
            proj_net.output_spec for proj_net in projection_networks
        ]
        output_spec = tf.nest.pack_sequence_as(action_spec,
                                               projection_distribution_specs)

        super(SevenWondersActorNetwork, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), output_spec=output_spec, name=name)

        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 2:
            raise ValueError('Only two actions is supported by this network')

        preprocessing_layers = {
            'age': tf.keras.layers.Reshape((1, ), name='actor/age'),
            'turn': tf.keras.layers.Reshape((1, ), name='actor/turn'),
            'players_coins': tf.keras.layers.Dense(5, name='actor/players_coins')
        }
        preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform')
        self._encoder = encoding_network.EncodingNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=False)

        self._projection_networks = projection_networks


    def call(self, observations, step_type=(), network_state=()):
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        states, network_state = self._encoder(
            observations, step_type=step_type, network_state=network_state)
        states = batch_squash.unflatten(states)

        outputs = [projection(states, outer_rank) for projection in self._projection_networks]
        output_actions = tf.nest.pack_sequence_as(self._action_spec, outputs)
        return output_actions, network_state


class SevenWondersValueNetwork(network.Network):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 name='SevenWondersActorNetwork'):
        super(SevenWondersValueNetwork, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name)

        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 2:
            raise ValueError('Only two actions is supported by this network')

        preprocessing_layers = {
            'age': tf.keras.layers.Reshape((1, ), name='value/age'),
            'turn': tf.keras.layers.Reshape((1, ), name='value/turn'),
            'players_coins': tf.keras.layers.Dense(5, name='value/players_coins')
        }
        preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform')
        self._encoder = encoding_network.EncodingNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=False)

        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003)

        self._value_projection_layer = tf.keras.layers.Dense(
            1,
            activation=tf.keras.activations.tanh,
            kernel_initializer=initializer,
            name='action')

    def call(self, observations, step_type=(), network_state=()):
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        state, network_state = self._encoder(observations, step_type=step_type, network_state=network_state)
        value = self._value_projection_layer(state)
        value = tf.reshape(value, [-1])
        value = batch_squash.unflatten(value)
        return value, network_state
