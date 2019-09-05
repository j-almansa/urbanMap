from cnn.ops import utils

def generate(allclasses,perclass,bounds,dims,mode,predictclass):
    utils.mklabel(allclasses,perclass,bounds,dims[0],mode,predictclass)
    utils.mkpatch(".\\data\\"+mode+"\\labels\\"+mode+"patchlist.csv", perclass)





if __name__ == "__main__":
    #import sys
    #pipeline(sys.argv)
    
    allclasses = [2,3,4,9,10]
    bounds = {2 :{"low":0.1,"upp":0.6},
              3 :{"low":0.2,"upp":0.4},
              4 :{"low":0.2,"upp":0.4},
              9 :{"low":0.1,"upp":0.5},
              10:{"low":0.2,"upp":0.4}}
    dims = [224,224,3]
    # Only one option below should be left uncommented
    # training option
    #perclass = 3
    #mode = "train"
    # testing option
    predictclass = [3] # must be singleton
    perclass = 1 # must be 1
    mode = "test"
    
    
    generate(allclasses,perclass,bounds,dims,mode,predictclass)
    