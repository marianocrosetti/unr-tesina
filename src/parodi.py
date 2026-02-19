"""
Parodi Pipeline for Pairwise Offline Signature Verification.

This module adapts the circular grid feature extraction from Parodi & Gómez (2010, 2011)
to a pairwise signature verification task. Instead of training per-author classifiers,
we train a single classifier to determine if two signatures belong to the same author.

Pipeline stages:
  1. Preprocessing: Otsu binarization with Gaussian blur.
  2. Data augmentation: Affine transformations (rotation, scaling).
  3. Feature extraction: Circular grid features via FastGrid (see features_grid.py).
  4. Pair featurization: Convert pairs of feature vectors to single vectors (e.g., difference).
  5. Training: SVM or other classifiers with optional scaling.

References:
  - Parodi & Gómez (2010): "A circular grid-based feature extraction approach"
  - Parodi & Gómez (2011): "A circular grid-based rotation invariant feature extraction approach"
"""
from libraries_import_all import *

from features_grid import *
from grid_search import *
from methodology import *

def otsu_preprocess(img):
    """ Binarize PIL Image using Gaussian blur + Otsu thresholding, then invert. """
    _,np_img = cv2.threshold(cv2.GaussianBlur(np.array(img),(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    np_img = 255-np_img
    return Image.fromarray(np_img)

def data_augmentation(img, n_augmentation):
    """ Apply n_augmentation random affine transforms (rotation ±15°, scale 0.75-1.25). """
    transformation=iaa.Affine(rotate=(-15,15),scale={"x": (0.75, 1.25), "y": (0.75, 1.25)})
    return [  Image.fromarray(transformation.augment_image(np.array(img))) for i in range(n_augmentation) ]


def preproc_pipeline(out_path, get_x, f):
    """ Preprocess image f with Otsu binarization and save to out_path. """
    nf = Path(out_path)/Path(f).name
    img = otsu_preprocess(get_x(f)).save(nf)
    return nf

def feature_extract_pipeline(out_path, get_x, ndiv, f):
    """ Extract FastGrid features from image f and save as pickle to out_path. """
    nf = str(Path(out_path)/Path(f).name)[:-3]+'pkl'
    FastGrid(f, get_x, N_divisions=ndiv).save(nf)
    return nf

def load_features(ndiv):
    """ Load precomputed grid features (npy) and filenames (txt) for given ndiv. """
    FEATURES_PATH = './data/parodi_out/features/'+str(ndiv)
    fns = L()
    xs = L()
    for npy_fp in glob.glob(FEATURES_PATH+'/*.npy'):
        with open(npy_fp, 'rb') as f:
            xs.append(np.load(f))
        with open(npy_fp[:-3]+'txt', 'r') as f:
            fns.extend( L(f.readlines()).map(lambda x : x.rstrip()) )
    x = np.concatenate(xs)
    return fns,x

class GridFeatureGetter:
    """ Lookup precomputed grid features by filename. Maps filename -> feature vector. """
    def __init__(self, ndiv):
        self.fns,self.x=load_features(ndiv)
        self.fns=self.fns.map(lambda x : Path(x).name)
    def __call__(self,f): 
        if isinstance(f,str): f=Path(f)
        return self.x[self.fns.index(f.name)]

def get_decision_values(clf, x):
    """ Get classifier decision values: predict_proba[:,1] or decision_function. """
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(x)
        assert len(y_proba.shape)==2 and y_proba.shape[1]==2
        assert len(clf.classes_)==2 and (clf.classes_==[False,True]).all()
        y_proba = y_proba[:,1]
    else:
        y_proba = clf.decision_function(x)
        len(y_proba.shape)==1
    return y_proba

def normalize(scalers, features):
    """ Apply fitted scalers to each feature channel (axis 1) in-place. """
    for i in range(features.shape[1]):
        features[:,i,:]=scalers[i].transform(features[:,i,:])
    return features

def featurize_pairs(pairs, features):
    """ Convert (idx1, idx2, label) pairs to (X=diff features, Y=labels, X1, X2). """
    x1, x2, f = tuple(map(L,zip(*pairs)))
    X1, X2, Y = features[x1], features[x2], np.array(f)
    X = np.stack([X1,X2],axis=1)
    X = X1-X2
    X=X.reshape((X.shape[0],-1))
    return X,Y,X1, X2

def augment_x(x,K): 
    """ Generate up to K augmented filenames from original filename x. """
    return L(Path(str(x.name)[:-7] + str(i) + '_6g.PNG') for i in range(min(10,K)))

def augment_y(y,K): 
    """ Repeat label y for up to K augmented samples. """
    return L(y for i in range(min(10,K)))

# Grid search options for SVM-based Parodi experiments
PARODI_GRID_SEARCH_OPTIONS = {
    'FEATURE_MIX_METHOD' : ['FeatureObtainer_Concat', 'FeatureObtainer_Difference'],
    'FFT' : ['OFF', 'ON'],  # DFT for rotational invariance
    'N_DIVISIONS' : ['8','16','32','64', '128'],  # Circular grid sectors
    'DATASET_AUGMENTATION' : ['OFF', 'ON'],
    'SCALER_METHOD' : ['NONE', 'StandardScaler', 'QuantileTransformer', 'MinMaxScaler', 'Normalizer'],
}

# Grid search options for Random Forest experiments (no scaling needed)
RF_GRID_SEARCH_OPTIONS = {
    'FEATURE_MIX_METHOD' : ['FeatureObtainer_Concat', 'FeatureObtainer_Difference'],
    'FFT' : ['OFF', 'ON'],
    'N_DIVISIONS' : ['8','16','32','64', '128'],
    'DATASET_AUGMENTATION' : ['OFF', 'ON'],
    'SCALER_METHOD' : ['NONE'],
}

def split_model_threshold_train(all_train, threshold_train_cls_set):
    """ Split training set: some classes for model training, others for threshold tuning. """
    all_train_cls_set = set(all_train[1])

    model_train_cls_set = all_train_cls_set.difference(threshold_train_cls_set)

    is_model_train = all_train[1].map(lambda cls : cls in model_train_cls_set)
    model_train = all_train[0][is_model_train], all_train[1][is_model_train]
    threshold_train = all_train[0][~is_model_train], all_train[1][~is_model_train]

    print(f'# train signatures: {len(all_train[0])} | # train model signatures {len(model_train[0])} | # train threshold signatures {len(threshold_train[0])} | check {len(model_train[0])+len(threshold_train[0])}')
    return model_train,threshold_train

def augment_signatures(ds,K):
    """ Expand dataset (filenames, labels) by adding K augmented versions per sample. """
    return ds[0].map(partial(augment_x,K=K)).flatten(), ds[1].map(partial(augment_y,K=K)).flatten()

def parodi_configurate(config, all_train, threshold_train_cls_set,  valid, valid_pairs, augment_ratio=4):
    """
    Configure Parodi pipeline: load features, augment, scale, and featurize pairs.
    
    Returns (Xm, ym, Xth, yth, Xv, yv): featurized pairs for model train, threshold train, and validation.
    """
    global PARODI_GRID_SEARCH_OPTIONS
    assert config_is_well_typed(config, PARODI_GRID_SEARCH_OPTIONS)
    fn2x = GridFeatureGetter(int(config['N_DIVISIONS']))

    model_train, threshold_train = split_model_threshold_train(all_train, threshold_train_cls_set)
    
    if config['DATASET_AUGMENTATION']=='ON':
        model_train = augment_signatures(model_train, augment_ratio)
    
    model_train_features = np.array(model_train[0].map(fn2x))
    threshold_train_features = np.array(threshold_train[0].map(fn2x))
    valid_features = np.array(valid[0].map(fn2x))
    
    if config['SCALER_METHOD']!='NONE':
        scalers = [ eval(config['SCALER_METHOD'])() for _ in range(model_train_features.shape[1]) ]
        for i in range(model_train_features.shape[1]):
            scalers[i].fit(np.array(L(model_train[0]+threshold_train[0]).map(fn2x))[:,i,:])
        model_train_features = normalize(scalers,model_train_features)
        threshold_train_features = normalize(scalers,threshold_train_features)
        valid_features = normalize(scalers,valid_features)
    
    model_ds_pairs = make_pairs(model_train[1])
    threshold_ds_pairs = make_pairs(threshold_train[1])
    
    Xm,ym,_,_ = featurize_pairs(model_ds_pairs,model_train_features)
    Xth,yth,_,_ = featurize_pairs(threshold_ds_pairs,threshold_train_features)
    Xv,yv,_,_ = featurize_pairs(valid_pairs,valid_features)
    return Xm,ym,Xth,yth,Xv,yv
