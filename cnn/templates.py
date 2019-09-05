'''
  Dictionary of (possibly trained) cnn architectures, along with
  hyperparameters and saved checkpoints.
'''
from cnn.archits import pixelnet
from cnn.archits import vgg16

# 2DO: change parameters of vgg16goteborg for finetuning
#      use momentum with decay for pixelnet2.2; e.g.
#         tflearn.optimizers.Momentum(learning_rate=0.01, lr_decay=0.96, decay_step=100)
#         regression = regression(net, optimizer=momentum)
def fetch(modelname):
    dict = {"vgg16":(vgg16.net,
                    {"optimizer":"adam", "loss":"categorical_crossentropy", "rate":0.001},
                    ".\\cnn\\ckpts\\tflearn\\vgg16.tflearn"),
            "vgg16goteborg":(vgg16.net,
                             {"optimizer":"adam", "loss":"categorical_crossentropy", "rate":0.001},
                            ".\\logs\\ckp\\vgg16goteborg\\vgg16goteborg.ckpt"),
            "pixelnet2.2":(pixelnet.net,
                          {"optimizer":"momentum", "loss":"categorical_crossentropy", "rate":0.001},
                          "")
           }
    return dict[modelname]