"""
Network architecture for VGG16.
Output has been modified in order to return cnn codes.

Original architecture from Aymeric Damien
[ https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network_finetuning.py ]

License: unknown
         contact author
"""
#import tensorflow as tf
import tflearn

#tf.reset_default_graph()

def net(input, numclasses):
    x     = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1')
    feat1 = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x     = tflearn.max_pool_2d(feat1, 2, strides=2, name='maxpool1')

    x     = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    feat2 = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x     = tflearn.max_pool_2d(feat2, 2, strides=2, name='maxpool2')

    x     = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x     = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    feat3 = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x     = tflearn.max_pool_2d(feat3, 2, strides=2, name='maxpool3')

    x     = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x     = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    feat4 = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x     = tflearn.max_pool_2d(feat4, 2, strides=2, name='maxpool4')

    x     = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x     = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    feat5 = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x     = tflearn.max_pool_2d(feat5, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, numclasses, activation='softmax', scope='fc8',
                                restore=False)

    return x, [feat1, feat2, feat3, feat4, feat5]