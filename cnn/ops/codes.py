'''
  Operations on CNN codes.
'''
import numpy as np
import scipy as sci

def formatcodes(Tcodes,model,x,X):
    sess=model.session
    #code1,code2,code3,code4,code5 = sess.run(Tcodes,feed_dict = {x:X})
    #upcode2 = scizoom(code2, [1.0, 2.0, 2.0,1.0], order=1, prefilter=False)
    #upcode3 = scizoom(code3, [1.0, 4.0, 4.0,1.0], order=1, prefilter=False)
    #upcode4 = scizoom(code4, [1.0, 8.0, 8.0,1.0], order=1, prefilter=False)
    #upcode5 = scizoom(code5, [1.0,16.0,16.0,1.0], order=1, prefilter=False)
    #codestack = np.concatenate([code1,upcode2,upcode3,upcode4,upcode5], axis=3)
    codevals = sess.run(Tcodes,feed_dict = {x:X})
    upcodes = [codevals[0]]
    for i in range(len(codevals[1:])):
        c = sci.ndimage.zoom(codevals[i+1],[1.0,2.0**(i+1),2.0**(i+1),1.0], order=1, prefilter=False)
        upcodes.append(c) #CHECK!!!
    codestack = np.concatenate(upcodes, axis=3)
    
    # 2DO: add ReLU?
    
    return codestack

# 2DO: dimensionality reduction on codes PCA|t-SNE
#def dimreduce(...):
    
#    return ...