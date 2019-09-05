'''
  Operations on networks.
'''
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os

def getcodes(modeltemplate, inplace, numclasses):
    network    = modeltemplate[0]
    hyperparam = modeltemplate[1]
    ckpfile    = modeltemplate[2]
    
    out, codes = network(inplace, numclasses)
    
    regression = tflearn.regression(out,
                                    optimizer=hyperparam["optimizer"],
                                    loss=hyperparam["loss"],
                                    learning_rate=hyperparam["rate"],
                                    restore=False)
    
    model = tflearn.DNN(regression)
    
    if ckpfile:
        model.load(ckpfile,weights_only=True)
    
    return model, out, codes


def train(modeltemplate, inplace, X, Y, numclasses, numepochs, validpct, batchsize,taskname):
    network    = modeltemplate[0]
    hyperparam = modeltemplate[1]
    ckpfile    = modeltemplate[2]
    
    out, _ = network(inplace, numclasses)
    
    regression = tflearn.regression(out,
                                    optimizer=hyperparam["optimizer"],
                                    loss=hyperparam["loss"],
                                    learning_rate=hyperparam["rate"])
    
    ckptdir = ".\\logs\\ckp\\" + taskname
    tbddir = ".\\logs\\tbd"
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(tbddir, exist_ok=True)
    
    model = tflearn.DNN(regression,
                        checkpoint_path=ckptdir,
                        max_checkpoints=3,
                        tensorboard_verbose=1,
                        tensorboard_dir=tbddir)
    
    if ckpfile:
        model.load(ckpfile,weights_only=True)
    
    model.fit(X,Y,
              n_epoch=numepochs,
              validation_set=validpct,
              shuffle=True,
              show_metric=True,
              batch_size=batchsize,
              snapshot_epoch=False,
              snapshot_step=200,
              run_id=taskname)
    
    #model.save(".\\cnn\\ckpts\\tflearn\\" + prjname + ".ckpt")
    
    return model
