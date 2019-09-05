import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import image_preloader
from itertools import product
import numpy as np
from scipy.ndimage import zoom as scizoom
import matplotlib
import matplotlib.pyplot as plt
import os


tf.reset_default_graph()
#------------------------------------------------------------------------------
# [ STAGE 1: Feature Extraction from VGG16 ]
#------------------------------------------------------------------------------
# [ Load and prepare the data set for VGG16 ]
data_dir = ".\\temptests"

X, Y = image_preloader(data_dir, image_shape=(224, 224), mode='folder',
                       categorical_labels=True, normalize=True,
                       files_extension=['.jpg', '.png'], filter_channel=True)

# Input Preprocessing
img_prep = ImagePreprocessing()
#img_prep.add_featurewise_zero_center(mean=[123.68, 116.779, 103.939],
#                                     per_channel=True)
img_prep.add_featurewise_zero_center()
#img_prep.add_featurewise_stdnorm()

x = tflearn.input_data(shape=[None, 224, 224, 3], name='input',
                       data_preprocessing=img_prep)

# [ Define VGG16 ]
num_classes = 5

def vgg16(input, num_class):

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

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
                                restore=False)

    return feat1, feat2, feat3, feat4, feat5, x

Tcode1, Tcode2, Tcode3, Tcode4, Tcode5, softmax = vgg16(x, num_classes)

# Define the training procedure (optimizer and logits)
regression = tflearn.regression(softmax, optimizer='adam',
                                loss='categorical_crossentropy',
                                learning_rate=0.001, restore=False)

# Define model object
model = tflearn.DNN(regression,
                    max_checkpoints=3, tensorboard_verbose=0)

# Load pretrained parameters and finetune
#model_path = ".\\cnn\\ckpts\\tflearn"
#model_file = os.path.join(model_path, "vgg16.tflearn")
model_path = ".\\xper\\xper01\\logs\\ckp\\goteborgvgg16"
model_file = os.path.join(model_path, "vgg16goteborg.ckpt")
model.load(model_file, weights_only=True)

# [ Obtain Features, Upsample and Concatenate ]
sess=model.session
code1,code2,code3,code4,code5 = sess.run([Tcode1,Tcode2,Tcode3,Tcode4,Tcode5],feed_dict = {x:X})
upcode2 = scizoom(code2, [1.0, 2.0, 2.0,1.0], order=1, prefilter=False)
upcode3 = scizoom(code3, [1.0, 4.0, 4.0,1.0], order=1, prefilter=False)
upcode4 = scizoom(code4, [1.0, 8.0, 8.0,1.0], order=1, prefilter=False)
upcode5 = scizoom(code5, [1.0,16.0,16.0,1.0], order=1, prefilter=False)
cnncodes = np.concatenate([code1,upcode2,upcode3,upcode4,upcode5], axis=3)
print("Shape of code1: ", code1.shape, "]")
plt.figure(1)
plt.subplot(151)
plt.imshow(code1[0,:,:,0])
plt.axis('off')
plt.subplot(152)
plt.imshow(upcode2[0,:,:,0])
plt.axis('off')
plt.subplot(153)
plt.imshow(upcode3[0,:,:,0])
plt.axis('off')
plt.subplot(154)
plt.imshow(upcode4[0,:,:,0])
plt.axis('off')
plt.subplot(155)
plt.imshow(upcode5[0,:,:,0])
plt.axis('off')
#plt.savefig("cnncodes.jpg")
plt.show()

#plt.figure(1)
##plt.subplot(151)
#plt.imshow(code1[0,:,:,0], cmap="Greys_r")
#plt.axis('off')
#plt.show()
##plt.subplot(152)
#plt.imshow(upcode2[0,:,:,0], cmap="Greys_r")
#plt.axis('off')
#plt.show()
##plt.subplot(153)
#plt.imshow(upcode3[0,:,:,0], cmap="Greys_r")
#plt.axis('off')
#plt.show()
##plt.subplot(154)
#plt.imshow(upcode4[0,:,:,0], cmap="Greys_r")
#plt.axis('off')
#plt.show()
##plt.subplot(155)
#plt.imshow(upcode5[0,:,:,0], cmap="Greys_r")
#plt.axis('off')
##plt.savefig("cnncodes.jpg")
#plt.show()
STOPHERE


#------------------------------------------------------------------------------
# [ STAGE 2: Apply MLP on Extracted Features ]
#------------------------------------------------------------------------------
tf.reset_default_graph()
mlp_input = tflearn.input_data(shape=[None,224,224,1472], name='mlp_input')

# [ Prepare Pixel Labels ]
classes = [3] # pixel classes
K = 6        # number of classes plus null class
sampperk = 1 # number of samples per class
#N = 10       # (K-1) times sampperk

#labelfn = lambda k,n: ".\\data\\goteborg\\temptestlabels\\label" + str(k) + "_" + str(n) + ".jpg"
#labels = [ plt.imread(labelfn(i,j)) for (i,j) in list(product(classes,range(sampperk))) ]
labelfn = lambda k,n: ".\\data\\goteborg\\temptestlabels\\label" + str(k) + "_" + str(n) + ".npy"
labels = [ np.load(labelfn(i,j)) for (i,j) in list(product(classes,range(sampperk))) ]

OHlabels = np.empty( (len(labels),224,224,K) )
for i in range(len(labels)):
    #classes = [2,3,4,9,10]
    labelvec = np.reshape(labels[i],224**2)
    mask = np.in1d(labelvec, classes, invert=True)
    labelvec[mask] = 0
    labelvec[labelvec==2 ] = 1
    labelvec[labelvec==3 ] = 2
    labelvec[labelvec==4 ] = 3
    labelvec[labelvec==9 ] = 4
    labelvec[labelvec==10] = 5
    onehotlabelmat = tflearn.data_utils.to_categorical(labelvec,K)
    OHlabels[i,:,:,:] = np.reshape(onehotlabelmat,(224,224,K))

X2 = cnncodes
Y2 = OHlabels
print("Shape of X2: ", X2.shape)
print("Shape of Y2: ", Y2.shape)



# [ Define MLP ]
fullyconn1 = tflearn.conv_2d(mlp_input, 1024, 1, activation='relu', scope='mlp_fc1')
fullyconn2 = tflearn.conv_2d(fullyconn1, 1024, 1, activation='relu', scope='mlp_fc2')
out        = tflearn.conv_2d(fullyconn2, K, 1, activation='softmax', scope='mlp_fc3')
#print('shape of out: ', out.shape)

# Define the training procedure
dhinet = tflearn.regression(out, optimizer='adam',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

# Define model object
dhimodel = tflearn.DNN(dhinet, checkpoint_path='.\\checkpoints\\dhigras',
                    max_checkpoints=3, tensorboard_verbose=0,
                    tensorboard_dir=".\\logs\\tflearn")

dhimodel_file = os.path.join(".\\pretrained\\dhigras", "dhigrasmodel.tflearn")
dhimodel.load(dhimodel_file, weights_only=True)

print("passed! :-)")



#------------------------------------------------------------------------------
# [ STAGE 3: Predict ]
#------------------------------------------------------------------------------
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#dhisess = dhimodel.session
#outv = dhisess.run(out,feed_dict = {mlp_input:X2})
#pixclass = np.argmax(outv[0,:,:,:], axis=2)
#print("Shape of output val: ", outv.shape)
#print("Shape of prediction: ", pixclass.shape)

#outv2 = dhimodel.predict(X2)
#arrayoutv2 = np.asarray(outv2)
#pixclass2 = np.argmax(arrayoutv2[0,:,:,:], axis=2)
#print("Shape of outv2 (as array): ", arrayoutv2.shape)
#print("Shape of prediction2: ", pixclass2.shape)

#print("Type of prediction: ",type(pixclass))
#print("Type of prediction2: ",type(pixclass2))

#print("Are predictions equal: ", np.array_equal(pixclass,pixclass2))
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
print("Type of X2: ", type(X2))
outv = dhimodel.predict(X2)
outv = np.asarray(outv)
pixclass = np.argmax(outv[0,:,:,:], axis=2)

outcmap = matplotlib.colors.ListedColormap(['black','grey','red','blue','yellow','green'])
outbounds = [0,1,2,3,4,5,6]
outnorm = matplotlib.colors.BoundaryNorm(outbounds, outcmap.N)

#plt.imshow(pixclass, cmap=outcmap, norm=outnorm)
#plt.savefig("road3.jpg")
#plt.show()
predfn = labelfn(classes[0],0)
predfn = predfn.replace("temptestlabels","temptestpreds")
np.save(predfn,pixclass)

