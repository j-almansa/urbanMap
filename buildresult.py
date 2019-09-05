import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def result():
    img1 = plt.imread(".\\data\\test\\classes\\class_0\\patch_0.jpg")
    true = np.load(   ".\\data\\test\\labels\\label_0.npy")
    pred = np.load(   ".\\data\\test\\preds\\pred_0.npy")

    trueclasses = [2,3,4,9,10] # whithout null
    predclasses = [1,2,3,4,5] # without null

    numclasses = len(trueclasses)
    numels = true.shape[0] * true.shape[1]
    print(true.dtype)
    print(pred.dtype)
    print("unique in true: ", np.unique(true))
    print("unique in pred: ", np.unique(pred))


    vpred = np.reshape(pred,numels)
    vtrue = np.reshape(true,numels)

    C = np.zeros((numclasses,numclasses))
    row = 0
    col = 0
    for pk in predclasses:
        row = 0
        for tk in trueclasses:
            C[row,col] = np.sum(np.logical_and(vpred==pk,vtrue==tk))
            row = row + 1
        col = col + 1
    #plt.matshow(C)
    #plt.show()


    gtcmap = matplotlib.colors.ListedColormap(['black','grey','red','blue','yellow','green','black'])
    gtbounds = [0,2,3,4,9,10,11,18]
    gtnorm = matplotlib.colors.BoundaryNorm(gtbounds, gtcmap.N)

    predcmap = matplotlib.colors.ListedColormap(['black','grey','red','blue','yellow','green'])
    predbounds = [0,1,2,3,4,5,6]
    prednorm = matplotlib.colors.BoundaryNorm(predbounds, predcmap.N)


    plt.figure(1)
    plt.subplot(131)
    plt.imshow(img1)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(true, cmap=gtcmap, norm=gtnorm)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(pred, cmap=predcmap, norm=prednorm)
    plt.axis('off')
    plt.savefig(".\\data\\test\\result_0.jpg")
    #plt.show()


    # 2DO Calculate metrics
    # Let C be the confussion matrix C for K classes (excluding null class)
    #     +---- predictions ---->
    #     |
    #     |
    # reference
    #     |
    #     |
    #     v
    #
    # C = [cij]
    # ith row               Ci.
    # jth col               C.j
    #
    # overall accuracy      oa = sum(diag(C)) / sum(C)
    #
    #
    #
    # true positives        [tp_1,...,tp_k] = [c11,...,cKK] = diag(C)
    # kth false positive    fp_k = sum(C.k) - tp_k
    # kth false negative    fn_k = sum(Ck.) - tp_k
    #
    # kth correctness       cor_k = tp_k / (tp_k + fp_k)
    # kth completeness      com_k = tp_k / (tp_k + fn_k)
    #
    # kth F1 score          F1_k = 2 * ((cor_k*com_k) / (cor_k+com_k))
    #
    #
    #
    # Jaccard coeff         J(Pk,Rk) = |Pk ^ Rk|/|Pk v Rk|, with J(Pk,Rk)=1 if both Pk,Rk = emptyset
    # (or IoU)                       = tp_k / (tp_k + fp_k + fn_k)
    #
    #
    # To consider: weighting factor of class-presence within scene
    #              w_k = Ck. / sum(C)
    #

    #C = np.ma.array(C, mask=np.isnan(C))
    #C = np.ma.array(C, mask=(C==0))
    tp = np.diagonal(C)
    fp = np.sum(C, axis=0) - tp
    fn = np.sum(C, axis=1) - tp

    with np.errstate(invalid="ignore"):
        cor = tp/(tp+fp)
        com = tp/(tp+fn)
        F1 = 2*((cor*com)/(cor+com))
        J = tp/(tp+fp+fn)
        oa = np.sum(tp)/np.sum(C)

    #np.set_printoptions(threshold=np.inf)
    print("[overall accuracy:      %.2f]" % oa)
    print("[F1 scores:            ", " ".join("{:.2f}".format(k) for k in F1), "]" )
    print("[Jaccard coefficients: ", " ".join("{:.2f}".format(k) for k in J), "]" )





if __name__ == "__main__":
    #import sys
    #pipeline(sys.argv)
    
    result()

