'''
  Various utility operations on spectral images.
'''
import tflearn
import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import os
import re



def mklabel(allclasses, numsamps, bounds, winsize, mode,predictclass=None): # mode = train|test
    #with rasterio.open(".\data\\Classification_clip.tif") as src:
    with rasterio.open(".\\data\\map.tif") as src:
        profile = src.profile.copy()
        data = src.read(1)
        #print(src.profile)
        #print(src.indexes)
        print("[Shape of data: ", data.shape, "]")
    
    if mode == "test":
        classes = predictclass
    else:
        classes = allclasses
    for i in range(len(classes)):
        classID = classes[i]
        noclassval = 1
        winhalf = winsize//2
        lowbound = winsize**2 * bounds[classID]["low"]
        uppbound = winsize**2 * bounds[classID]["upp"]
        found = 0
        trial = 0
        maxiter = 10000
        
        while ( found < numsamps ) and ( trial <= maxiter ):
            trial = trial + 1
            col = np.random.randint(winhalf-1, data.shape[1]-(winhalf-1))
            row = np.random.randint(winhalf-1, data.shape[0]-(winhalf-1))
            startcol = col-(winhalf-1)
            untilcol = col+(winhalf+1)
            startrow = row-(winhalf-1)
            untilrow = row+(winhalf+1)
            clip = data[startcol:untilcol, startrow:untilrow]
            pixcount = np.sum((clip==classID).astype(int))
            
            #print("[untilrow-startrow: (%d-%d)=%d" % (untilrow,startrow,untilrow-startrow) )
            #badclip = ( clip[winhalf,winhalf] != classes[i] )             # center pixel does not belong to class
            #        | ( np.any(clip == noclassval) )                      # there are null pixels
            #        | ( (pixcount < lowbound) | (uppbound <= pixcount) )  # number of in-class pixels is out of bounds
            #print("clip shape: ", clip.shape)
            nocenter    = False #( clip[(untilcol-startcol)//2, (untilrow-startrow)//2] != classID )
            nullpixels  = ( np.any(clip == noclassval) )
            outofbounds = (pixcount < lowbound) | (uppbound <= pixcount)
            badclip = nocenter | nullpixels | outofbounds
            #if ( np.any(clip == noclassval) ) | ( (pixcount < lowbound) | (uppbound <= pixcount) ):
            if badclip:
                continue

            # make no-class label equal to zero
            clip = np.reshape(clip,winsize**2)
            mask = np.in1d(clip, allclasses, invert=True)
            clip[mask] = 0
            #print("Are there any in-class pixels: ", np.any( np.in1d(clip, classes) ))
            #print("Are there any out-class pixels: ", np.any( np.in1d(clip, [1,5,6,7,8,11,12,13,14,15,16,17,18]) ))
            #print("Are there any zeros: ", np.any( np.in1d(clip, [0]) ))
            clip = np.reshape(clip,(winsize,winsize))

            # save label as numpy array
            basefn = ".\\data\\" + mode + "\\labels\\label_" + str(i*numsamps + found)
            dirname = os.path.dirname(basefn)
            os.makedirs(dirname, exist_ok=True)
            
            #gtcmap = matplotlib.colors.ListedColormap(['black','grey','red','blue','yellow','green','black'])
            #gtbounds = [0,2,3,4,9,10,11,18]
            #gtnorm = matplotlib.colors.BoundaryNorm(gtbounds, gtcmap.N)
            #plt.imshow(clip, cmap=gtcmap, norm=gtnorm)
            #plt.show()
            np.save(basefn + ".npy",clip)
            
            # save pixel region info on csv file, to extract corresponding
            # patches from multispectral image.
            with open(".\\data\\" + mode + "\\labels\\" + mode + "patchlist.csv", 'a', newline='') as fp:
                writer = csv.writer(fp, delimiter=',')
                writer.writerow([basefn,startcol,untilcol,startrow,untilrow])
            
            # show region with class emphasized (black)
    #        classcmap = matplotlib.colors.ListedColormap(['gray','black','gray'])
    #        classbounds = [0,classID,classID+1,18]
    #        classnorm = matplotlib.colors.BoundaryNorm(classbounds, classcmap.N)
    #        plt.imshow(clip, cmap=classcmap, norm=classnorm)
    #        plt.show()
            
            found = found + 1
            if found == numsamps:
                print("[Class: %d][Trial: %d][Found: %d]" % (classID,trial,found))
    print("labels done")


    #img = plt.imread(basefn + ".jpg", format='jpeg')
    #print('shape of img: ', np.ndim(img))
    #plt.imshow(img, cmap='Greys_r')
    #plt.show()



def mkpatch(patchcsv,patchesperclass):
    with rasterio.open(".\\data\\img.tif") as src:
        profile = src.profile.copy()
        data = src.read()
        #print("[DType band 1: %s]" % src.dtypes[0])
        #print("[height and width: (%d,%d)]"% (src.height, src.width) )
        #print("[CRS: ", src.crs, "]")
        #print("[Transform: ", src.transform, "]")
        #print("[bands: ", src.count, "][Indexes: ", src.indexes, "]")
        print("[Shape of data: ", data.shape, "]")
        

    
    with open(patchcsv,'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        i = 0
        for line in reader:
            basefn = line[0]
            startcol = int(line[1])
            untilcol = int(line[2])
            startrow = int(line[3])
            untilrow = int(line[4])
            clip = data[:, startcol:untilcol, startrow:untilrow]
            #print("Shape of clip: ", clip.shape, "]")


            patchfn = re.sub(r"\\labels\\label_\d+", "\\classes\\class_"+str(i//patchesperclass)+"\\patch_"+str(i%patchesperclass), basefn)
            dirname = os.path.dirname(patchfn)
            os.makedirs(dirname, exist_ok=True)
            
            with rasterio.open(patchfn + ".jpg", 'w',
                               driver='JPEG',
                               dtype='uint8',
                               width=224,
                               height=224,
                               count=src.count,
                               crs=profile['crs'],
                               transform=profile['transform'],
                               nodata=profile['nodata']) as dst:
                dst.write(clip)
            
            os.remove(patchfn+".jpg.aux.xml")
            #clip = np.swapaxes(clip,0,1)
            #clip = np.swapaxes(clip,1,2)
            #print("Shape of clip: ", clip.shape, "]")
            #print("shape of clip", clip.shape)
            #if i == 0:
            #    plt.imshow(clip)
            #    plt.show()
            i = i + 1
    print("patches done")



def OHpixlabels(classes, sampperk, pixclasses, size, mode):
    K = pixclasses # K = total num. of classes plus one, to include a null class
    labels = [ np.load( ".\\data\\" + mode + "\\labels\\label_" + str(i) + ".npy" ) for i in range(len(classes)*sampperk) ]
    
    OHlabels = np.empty( (len(labels),size,size,K) )
    for i in range(len(labels)):
        labelvec = np.reshape(labels[i],size**2)
        mask = np.in1d(labelvec, classes, invert=True)
        labelvec[mask] = 0
        #labelvec[labelvec==2 ] = 1
        #labelvec[labelvec==3 ] = 2
        #labelvec[labelvec==4 ] = 3
        #labelvec[labelvec==9 ] = 4
        #labelvec[labelvec==10] = 5
        for j in range(len(classes)):
            labelvec[labelvec==classes[j]] = j+1
        onehotlabelmat = tflearn.data_utils.to_categorical(labelvec,K)
        OHlabels[i,:,:,:] = np.reshape(onehotlabelmat,(size,size,K))
    return OHlabels

# 2DO: normalize image data zero-mean + stddev (+...?)
#def normalize(...):
    
#    return ...