"""
Network architecture for PixelNet.
Only contains the MLP part.
"""
#import tensorflow as tf
import tflearn


#tf.reset_default_graph()

def net(input, numclasses):
    x     = tflearn.conv_2d(input, 1024      , 1, activation='relu'   , scope='mlp_fc1')
    x     = tflearn.conv_2d(x    , 1024      , 1, activation='relu'   , scope='mlp_fc2')
    x     = tflearn.conv_2d(x    , numclasses, 1, activation='softmax', scope='mlp_fc3')

    return x, []