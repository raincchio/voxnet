def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = tf.reduce_mean(input_x, [1, 2, 3, 4])

        excitation = tf.nn.relu(tf.layers.dense(squeeze, use_bias=False, units=7))
        excitation = tf.nn.sigmoid(tf.layers.dense(excitation, units=7))

        excitation = tf.reshape(excitation, [-1, 1, 1, 1, 7])

        scale = input_x * excitation

        return scale