import tensorflow as tf

def NonLocalBlock(x, out_channels, sub_sample=True, is_bn=True, scope='NonLocalBlock'):
    mid_channels = x.shape()[-1]

    g = tf.layers.conv3d(x, filters=mid_channels, kernel_size=1, strides=1, padding="same",
                     kernel_initializer=self.init)
    phi = tf.layers.conv3d(x, filters=16, kernel_size=1, strides=1, padding="same",
                         kernel_initializer=self.init)
    theta = tf.layers.conv3d(x, filters=16, kernel_size=1, strides=1, padding="same",
                         kernel_initializer=self.init)


    g = tf.reshape(g, [batchsize,mid_channels, -1])
    g = tf.transpose(g, [0,2,1])

    theta = tf.reshape(theta, [batchsize, mid_channels, -1])
    theta = tf.transpose(theta, [0,2,1])
    phi = tf.reshape(phi, [batchsize, mid_channels, -1])

    f = tf.matmul(theta, phi)
    # ???
    f = tf.nn.softmax(f, -1)
    y = tf.matmul(f, g)
    y = tf.reshape(y, [-1, 64, 64, 64, mid_channels])

    y = tf.layers.conv3d(y, filters=x.shape()[-1], kernel_size=1, strides=1, padding="same",
                     kernel_initializer=self.init)
    z = x + y
    return tf.nn.batch_normalization(z,training = self)