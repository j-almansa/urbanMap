from cnn.ops import utils
from cnn.ops import nets
from cnn.ops import codes
from cnn import templates
import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import image_preloader
import numpy as np



def combine(prjname,basenet,classes,perclass,dims,withtransfer):
    # load data
    tf.reset_default_graph()
    X1, Y1 = image_preloader(".\\data\\train\\classes",
                            image_shape=tuple(dims)[0:2],
                            mode="folder",
                            categorical_labels=True,
                            normalize=True,
                            files_extension=[".jpg",".png"],
                            filter_channel=True)
    
    img_prep = ImagePreprocessing()
    # 2FIX: normalization should be done per channel
    img_prep.add_featurewise_zero_center()
    #img_prep.add_featurewise_stdnorm()
    
    x1 = tflearn.input_data(shape=np.insert(np.asarray(dims, dtype=object),0,None).tolist(),
                            data_preprocessing=img_prep,
                            name="input")
    
    # extract codes
    if withtransfer:
        checkpoint = "vgg16goteborg"
    else:
        checkpoint = "vgg16"
    
    mtnew = templates.fetch(checkpoint)
    
    
    # 2DO: is bilinear interpolation enough?
    numclasses = len(classes)
    model1new, _, codelist = nets.getcodes(mtnew, x1, numclasses)
    
    # prepare input for second stage
    # 2DO: should X2 be normalized? is ReLUing sufficient?
    pixclasses = numclasses + 1
    X2 = codes.formatcodes(codelist,model1new,x1,X1)
    Y2 = utils.OHpixlabels(classes, perclass, pixclasses, dims[0], "train")
    
    # train second net on codes
    tf.reset_default_graph()

    x2 = tflearn.input_data(shape=np.insert(np.asarray(list(X2.shape)[1:], dtype=object),0,None).tolist(),
                            name='mlp_input')
    
    mt2 = templates.fetch("pixelnet2.2")
    
    
    numepochs = 1
    validpct = 0.1
    batchsize = 3
    print("numclasses: %d" % pixclasses)
    print("shape of x2: ", x2.shape)
    print("shape of X2: ", X2.shape)
    model2 = nets.train(mt2,
                        x2, X2, Y2,
                        pixclasses,
                        numepochs,
                        validpct,
                        batchsize,
                        "pixelnet" + prjname)
    
    # save model
    model2.save(".\\logs\\ckp\\" + "pixelnetgoteborg\\" + "pixelnet" + prjname + ".ckpt")
    print("bien!")





if __name__ == "__main__":
    #import sys
    #pipeline(sys.argv)
    
    classes = [2,3,4,9,10]
    bounds = {2 :{"low":0.1,"upp":0.6},
              3 :{"low":0.2,"upp":0.4},
              4 :{"low":0.2,"upp":0.4},
              9 :{"low":0.1,"upp":0.5},
              10:{"low":0.2,"upp":0.4}}
    perclass = 3
    dims = [224,224,3]
    withtransfer = False
    
    combine("goteborg","vgg16",classes,perclass,dims,withtransfer)