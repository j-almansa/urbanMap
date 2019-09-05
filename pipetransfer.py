from cnn.ops import nets
from cnn import templates
import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import image_preloader
import numpy as np

def transfer(prjname,basenet,numclasses,dims):
    # load data
    tf.reset_default_graph()
    X1, Y1 = image_preloader(".\\data\\train\\classes",
                            image_shape=tuple(dims)[0:2],
                            mode="folder",
                            categorical_labels=True,
                            normalize=True,
                            files_extension=[".jpg",".png"],
                            filter_channel=True)
    
    # retrain basenet on new dataset
    img_prep = ImagePreprocessing()
    # 2FIX: normalization should be done per class and per channel
    img_prep.add_featurewise_zero_center()
    #img_prep.add_featurewise_stdnorm()
    
    x1 = tflearn.input_data(shape=np.insert(np.asarray(dims, dtype=object),0,None).tolist(),
                            data_preprocessing=img_prep,
                            name="input")
    mt1 = templates.fetch("vgg16")
    
    numepochs = 2
    validpct = 0.1
    batchsize = 3
    model1 = nets.train(mt1,
                       x1, X1, Y1,
                       numclasses,
                       numepochs,
                       validpct,
                       batchsize,
                       "vgg16" + prjname)
    
    model1.save(".\\logs\\ckp\\" + "vgg16goteborg\\" + "vgg16" + prjname + ".ckpt")
    
#     NB:!!!!!! saved checkpoint filename is appended .data-00000-of-00001
#               by the system; full name is vgg16goteborg.ckpt.data-00000-of-00001 !!!!!!
#     2DO: modify dict in template to include retrained vgg16 with goteborg data.
#          at the moment, changes are hard-coded





if __name__ == "__main__":
    #import sys
    #pipeline(sys.argv)
    
    numclasses = 5
    dims = [224,224,3]
    
    transfer("goteborg","vgg16",numclasses,dims)