from cnn.ops import utils
from cnn.ops import nets
from cnn.ops import codes
from cnn import templates
import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import image_preloader
import numpy as np
import os

def predict(prjname,basenet,predictclass,perclass,numclasses,dims,withtransfer):
    
    # load data
    tf.reset_default_graph()
    X1, Y1 = image_preloader(".\\data\\test\\classes",
                            image_shape=tuple(dims)[0:2],
                            mode="folder",
                            categorical_labels=True,
                            normalize=True,
                            files_extension=[".jpg",".png"],
                            filter_channel=True)
    
    # retrain vgg16 on goteborg
    img_prep = ImagePreprocessing()
    # 2FIX: normalization should be done per channel
    img_prep.add_featurewise_zero_center()
    #img_prep.add_featurewise_stdnorm()
    
    x1 = tflearn.input_data(shape=np.insert(np.asarray(dims, dtype=object),0,None).tolist(),
                            data_preprocessing=img_prep,
                            name="input")
    
    if withtransfer:
        checkpoint = "vgg16goteborg"
    else:
        checkpoint = "vgg16"
        
    mt1 = templates.fetch(checkpoint)
    #numclasses = 5
    model1, _, codelist = nets.getcodes(mt1, x1, numclasses)
    
    # prepare input for second stage
    # 2DO: should X2 be normalized? is ReLUing sufficient?
    pixclasses = numclasses + 1
    X2 = codes.formatcodes(codelist,model1,x1,X1)
    Y2 = utils.OHpixlabels(predictclass, perclass, pixclasses, dims[0], "test")
    
    # use second net on codes
    tf.reset_default_graph()

    x2 = tflearn.input_data(shape=np.insert(np.asarray(list(X2.shape)[1:], dtype=object),0,None).tolist(),
                            name='mlp_input')
    
    mt2 = templates.fetch("pixelnet2.2")
    model2, _, _ = nets.getcodes(mt2, x2, pixclasses)
    model2.load(".\\logs\\ckp\\" + "pixelnetgoteborg\\" + "pixelnet" + prjname + ".ckpt",weights_only=True)

    outv = model2.predict(X2)
    outv = np.asarray(outv)
    pixclass = np.argmax(outv[0,:,:,:], axis=2)
    
    preddir = ".\\data\\test\\preds"
    os.makedirs(preddir, exist_ok=True)
    np.save( preddir + "\\" "pred_0.npy",pixclass )
    print("bien!")




if __name__ == "__main__":
    #import sys
    #pipeline(sys.argv)
    
    #classes = [2,3,4,9,10]
    #bounds = {2 :{"low":0.1,"upp":0.6},
    #          3 :{"low":0.2,"upp":0.4},
    #          4 :{"low":0.2,"upp":0.4},
    #          9 :{"low":0.1,"upp":0.5},
    #          10:{"low":0.2,"upp":0.4}}
    
    predictclass = [3]
    perclass = 1
    numclasses = 5
    dims = [224,224,3]
    withtransfer = False
    
    predict("goteborg","vgg16",predictclass,perclass,numclasses,dims,withtransfer)