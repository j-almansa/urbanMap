import pipedata
import pipetransfer
import pipecombine
import pipepredict

def line(prjname,basenet,allclasses,perclass,bounds,dims,predictclass,withtransfer):
    numclasses = len(allclasses)
    pipedata.generate(allclasses,perclass,bounds,dims,"train",[])
    if withtransfer:
        pipetransfer.transfer(prjname,basenet,numclasses,dims)
    pipecombine.combine(prjname,basenet,allclasses,perclass,dims,withtransfer)
    pipedata.generate(allclasses,1,bounds,dims,"test",predictclass)
    pipepredict.predict(prjname,basenet,predictclass,1,numclasses,dims,withtransfer)
    buildresult()





if __name__ == "__main__":
    #import sys
    #pipeline(sys.argv)
    
    allclasses = [2,3,4,9,10]
    predictclass = [3]
    bounds = {2 :{"low":0.1,"upp":0.6},
              3 :{"low":0.2,"upp":0.4},
              4 :{"low":0.2,"upp":0.4},
              9 :{"low":0.1,"upp":0.5},
              10:{"low":0.2,"upp":0.4}}
    perclass = 5
    dims = [224,224,3]
    basenet = "vgg16"
    withtransfer = False
    prjname = "goteborg"
    
    line(prjname,basenet,allclasses,perclass,bounds,dims,predictclass,withtransfer)
    