import tensorflow as tf


class SparseVoxNet_DSC:
    def __init__(self, n_classes=3, dropout_rate=0.2, is_training=True):
        self.init = tf.truncated_normal_initializer(stddev=0.01)
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.is_training = is_training

    def __call__(self, x):
        with tf.name_scope("conv1"):
            conv_preliminary = tf.layers.conv3d(x, filters=16, kernel_size=3, strides=2, padding="same",
                                                kernel_initializer=self.init)
        with tf.name_scope("non_local_block"):
            spatial = self.non_local_block(conv_preliminary)
        with tf.name_scope("dense1"):
             dense1 = self.dense(spatial)
        with tf.name_scope("se_block1"):
            scale1 = self.squeeze_excitation(dense1)
        with tf.name_scope("transform1"):
            transform_conv1, transform_pooling1 = self.transform(scale1, n_filters=160)
        with tf.name_scope("dense2"):
            dense2 = self.dense(transform_pooling1)
        with tf.name_scope("se_block2"):
            scale2 = self.squeeze_excitation(dense2)
        with tf.name_scope("bn26"):
            transform_conv2, transform_pooling2 = self.transform(scale2, n_filters=304, final_layer=True)
        with tf.name_scope("deconv1"):
            deconv1 = tf.layers.conv3d_transpose(transform_conv2, filters=128, kernel_size=4, strides=2, use_bias=False,
                                                 padding="same", kernel_initializer=self.init)
        with tf.name_scope("deconv2"):
            deconv2 = tf.layers.conv3d_transpose(deconv1, filters=64, kernel_size=4, strides=2, use_bias=False,
                                                 padding="same", kernel_initializer=self.init)
        with tf.name_scope("logit1"):
            logit1 = tf.layers.conv3d(deconv2, filters=self.n_classes, kernel_size=1, padding="same",
                                      kernel_initializer=self.init)

        with tf.name_scope("deconv_aux"):
            deconv_aux = tf.layers.conv3d_transpose(transform_conv1, filters=64, kernel_size=4, strides=2,
                                                    use_bias=False, padding="same", kernel_initializer=self.init)

        with tf.name_scope("logit2"):
            logit2 = tf.layers.conv3d(deconv_aux, filters=self.n_classes, kernel_size=1, strides=1,
                                      padding="same", kernel_initializer=self.init)

        prob_map = tf.nn.softmax(logit1, axis=4)
        annotation = tf.argmax(logit1, axis=4, name="prediction")

        return logit1, logit2, prob_map, annotation

    def non_local_block(self, x, mid_channels=8):

        theta = tf.layers.conv3d(x, filters=8, kernel_size=1, strides=1, padding="same",
                                 kernel_initializer=self.init)
        phi = tf.layers.conv3d(x, filters=8, kernel_size=1, strides=2, padding="valid",
                               kernel_initializer=self.init)
        g = tf.layers.conv3d(x, filters=8, kernel_size=1, strides=2, padding="valid",
                             kernel_initializer=self.init)

        theta = tf.reshape(theta, [-1, 4096, 8])
        phi = tf.reshape(phi, [-1, 512, 8])
        phi = tf.transpose(phi, [0, 2, 1])

        g = tf.reshape(g, [-1, 512, 8])

        f = tf.matmul(theta, phi)
        # ???
        f = tf.nn.softmax(f, -1)
        y = tf.matmul(f, g)
        y = tf.reshape(y, [-1, 16, 16, 16, mid_channels])

        y = tf.layers.conv3d_transpose(y, filters=16, kernel_size=1, strides=2, padding="same",
                             kernel_initializer=self.init)
        z = x + y
        return self.batch_normalization(z, training=self.is_training)


    def dense(self, x, growth_rate=12):
        # 1
        bn = tf.nn.relu(x)
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, padding="same", kernel_initializer=self.init)
        drop1 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        # 2
        bn = tf.nn.relu(self.batch_normalization(drop1, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, padding="same", kernel_initializer=self.init)
        drop2 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        # 3
        bn = tf.nn.relu(self.batch_normalization(drop2, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, padding="same", kernel_initializer=self.init)
        drop3 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        # 4
        bn = tf.nn.relu(self.batch_normalization(drop3, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, padding="same", kernel_initializer=self.init)
        drop4 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        # 5
        bn = tf.nn.relu(self.batch_normalization(drop4, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, dilation_rate=2, padding="same", kernel_initializer=self.init)
        drop5 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        # 6
        bn = tf.nn.relu(self.batch_normalization(drop5, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, dilation_rate=3, padding="same", kernel_initializer=self.init)
        drop6 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        # 7
        bn = tf.nn.relu(self.batch_normalization(drop6, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, dilation_rate=5, padding="same", kernel_initializer=self.init)
        drop7 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)

        # dense = tf.concat([tf.expand_dims(i, -1) for i in [drop7, drop6, drop5, drop4, drop3, drop2, drop1]], axis=5)
        dense = tf.concat([drop7, drop6, drop5, drop4, drop3, drop2, drop1, x], axis=4)
        return dense

    def transform(self, x, n_filters, final_layer=False):
        bn = tf.nn.relu(self.batch_normalization(x, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=n_filters, kernel_size=1, padding="valid", kernel_initializer=self.init)

        if not final_layer:
            drop = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
            pool = tf.layers.max_pooling3d(drop, pool_size=2, strides=2)

        return conv, pool

    def squeeze_excitation(self, dense):

        squeeze_d = tf.reduce_mean(dense, [1, 2, 3, 4])
        # squeeze_x = tf.reduce_mean(x, [1, 2, 3, 4])
        excitation = tf.concat([squeeze_d], 1)

        excitation = tf.nn.relu(tf.layers.dense(excitation, use_bias=False, units=7))
        excitation = tf.nn.sigmoid(tf.layers.dense(excitation, units=7))

        excitation = tf.reshape(excitation, [-1, 1, 1, 1, 1, 7])

        scale_dense = dense * excitation[:, :, :, :, :, 0:6]
        # scale_x = tf.squeeze(x * excitation[:, :, :, :, :, 7:8], -1)
        scale_dense = tf.concat([tf.squeeze(scale_dense[:, :, :, :, :, i:i+1], -1) for i in range(6)], axis=4)

        return tf.concat([scale_dense, x], axis=4)

    def batch_normalization(self, x, training):
        training = tf.constant(training)
        depth = x.get_shape()[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed_tensor = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed_tensor
####################