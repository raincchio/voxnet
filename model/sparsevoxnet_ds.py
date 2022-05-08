import tensorflow as tf


class SparseVoxNet_DS:
    def __init__(self, n_classes=3, dropout_rate=0.2, is_training=True):
        self.init = tf.truncated_normal_initializer(stddev=0.01)
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.is_training = is_training

    def __call__(self, x):
        with tf.name_scope("conv1s"):
            conv_preliminary = tf.layers.conv3d(x, filters=16, kernel_size=3, strides=2, padding="same",
                                                kernel_initializer=self.init)
        with tf.name_scope("non_locals"):
            spatial = self.non_local_block(conv_preliminary)
        with tf.name_scope("concat12s"):
            feature1 = self.dense(spatial)
        with tf.name_scope("pooling1s"):
            conv14, pooling1 = self.transform(feature1, n_filters=100)
        with tf.name_scope("concat24s"):
            feature2 = self.dense(pooling1)
        with tf.name_scope("bn26s"):
            bn26 = tf.nn.relu(self.batch_normalization(feature2, training=self.is_training))
        with tf.name_scope("conv27s"):
            conv27 = tf.layers.conv3d(bn26, filters=184, kernel_size=1, padding="valid", kernel_initializer=self.init)
        with tf.name_scope("deconv1s"):
            deconv1 = tf.nn.relu(
                tf.layers.conv3d_transpose(conv27, filters=128, kernel_size=4, strides=2, use_bias=False,
                                           padding="same",
                                           kernel_initializer=self.init))
        with tf.name_scope("deconv2s"):
            deconv2 = tf.nn.relu(
                tf.layers.conv3d_transpose(deconv1, filters=64, kernel_size=4, strides=2, use_bias=False,
                                           padding="same",
                                           kernel_initializer=self.init))
        with tf.name_scope("logit1s"):
            logit1 = tf.layers.conv3d(deconv2, filters=self.n_classes, kernel_size=1, padding="same",
                                      kernel_initializer=self.init)
        # skip link
        with tf.name_scope("score_aux1s"):
            socre_aux1 = tf.layers.conv3d(conv14, filters=self.n_classes, kernel_size=1, padding="same",
                                          kernel_initializer=self.init)

        with tf.name_scope("logit2s"):
            logit2 = tf.layers.conv3d_transpose(socre_aux1, filters=self.n_classes, kernel_size=4, strides=2,
                                                use_bias=False,
                                                padding="same", kernel_initializer=self.init)

        with tf.name_scope('feature1_distilation_deconv2s'):
            feature1_distilation = tf.layers.conv3d_transpose(feature1, filters=64, kernel_size=4, strides=2,
                                                use_bias=False,
                                                padding="same", kernel_initializer=self.init)
            feature1_distilation_logit = tf.layers.conv3d(feature1_distilation, filters=self.n_classes, kernel_size=1, padding="same",
                                          kernel_initializer=self.init)
        with tf.name_scope('feature2_distilation_deconv2s'):
            feature2_distilation = tf.layers.conv3d_transpose(feature2, filters=64, kernel_size=4, strides=4,
                                                              use_bias=False,
                                                              padding="same", kernel_initializer=self.init)
            feature2_distilation_logit = tf.layers.conv3d(feature2_distilation, filters=self.n_classes, kernel_size=1, padding="same",
                                                           kernel_initializer=self.init)

        prob_map = tf.nn.softmax(logit1, axis=4)
        annotation = tf.argmax(logit1, axis=4, name="prediction")

        # annotation_f1 = tf.argmax(feature1_distilation_logit, axis=4, name="prediction_f1")
        # annotation_f2 = tf.argmax(feature2_distilation_logit, axis=4, name="prediction_f2")
        feature_t = tf.stop_gradient(tf.identity(deconv2))

        return logit1, logit2, prob_map, annotation, feature1_distilation, feature2_distilation, feature_t, feature1_distilation_logit, feature2_distilation_logit

    def non_local_block(self, x, mid_channels=8):

        g = tf.layers.conv3d(x, filters=mid_channels, kernel_size=1, strides=2, padding="same",
                             kernel_initializer=self.init)
        phi = tf.layers.conv3d(x, filters=8, kernel_size=1, strides=2, padding="same",
                               kernel_initializer=self.init)
        theta = tf.layers.conv3d(x, filters=8, kernel_size=1, strides=2, padding="same",
                                 kernel_initializer=self.init)

        g = tf.reshape(g, [-1, mid_channels, 16*16*16])
        g = tf.transpose(g, [0, 2, 1])

        theta = tf.reshape(theta, [-1, mid_channels, 16*16*16])
        theta = tf.transpose(theta, [0, 2, 1])
        phi = tf.reshape(phi, [-1, mid_channels, 16*16*16])

        f = tf.matmul(theta, phi)
        # ???
        f = tf.nn.softmax(f, -1)
        y = tf.matmul(f, g)
        y = tf.reshape(y, [-1, 16, 16, 16, mid_channels])

        y = tf.layers.conv3d_transpose(y, filters=16, kernel_size=4, strides=2, padding="same", kernel_initializer=self.init)
        z = x + y
        return z

    def dense(self, x, growth_rate=12):
        # 1
        bn = tf.nn.relu(self.batch_normalization(x, training=self.is_training))
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
        #tmp
        bn = tf.nn.relu(self.batch_normalization(drop4, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, padding="same", kernel_initializer=self.init)
        drop5 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        # 5
        bn = tf.nn.relu(self.batch_normalization(drop5, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, dilation_rate=2, padding="same", kernel_initializer=self.init)
        drop6 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        # 6
        bn = tf.nn.relu(self.batch_normalization(drop6, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, dilation_rate=3, padding="same", kernel_initializer=self.init)
        drop7 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        # 7
        bn = tf.nn.relu(self.batch_normalization(drop7, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, dilation_rate=5, padding="same", kernel_initializer=self.init)
        drop8 = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)

        return tf.concat([drop8, drop7, drop6, drop5, drop4, drop3, drop2, drop1, x], axis=4)

    def transform(self, x, n_filters):
        bn = tf.nn.relu(self.batch_normalization(x, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=n_filters, kernel_size=1, padding="valid", kernel_initializer=self.init)
        drop = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        pool = tf.layers.max_pooling3d(drop, pool_size=2, strides=2)

        return conv, pool

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

####################