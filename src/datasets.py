"""
Dataset loaders for signature and digit matching tasks.

This module provides functions to load various signature and handwriting datasets
including SIGCOMP 2009, MNIST, Omniglot, and CEDAR. It also provides convenience
functions to create paired datasets for Siamese network training.
"""

from src.all_deps import *
from src.models import y_transform_category, x_transform_signatures, x_transform_mnist
from src.visualization import show_pair
from src.pair_dataset import PairDataset

# Sigcomp 2009

SIGCOMP2009_FILE_PATHS = ['./data/sigcomp2009/NISDCC', './data/sigcomp2009/NFI']

SIGCOMP2009_REGEXES = [ r'NISDCC-([0-9]{3})_([0-9]{3})_([0-9]{3})_6g.', r'NFI-([0-9]{3})([0-9]{2})([0-9]{3}).' ]

SIGCOMP2009_REGEX_Y_GROUP = [ 2, 3 ]

SIGCOMP2009_CLASS_TAGS = [ 'NISDCC-', 'NFI-' ]
    
def sigcomp2009(only_orignals=True, train=True):
    """
    Load SIGCOMP 2009 signature dataset.
    
    Args:
        only_orignals (bool): If True, only load original signatures (not forgeries). Defaults to True.
        train (bool): If True, load training set (NISDCC), otherwise load validation set (NFI). Defaults to True.
    
    Returns:
        list: List of tuples (file_path, class_label) for each signature image.
    """
    def match(fp, i): 
        return re.match(SIGCOMP2009_REGEXES[train], os.path.basename(fp)).group(i)
            
    files = [f for f in glob.glob(SIGCOMP2009_FILE_PATHS[train]+'/*') if f.endswith('png') or f.endswith('PNG') ]
    if only_orignals:
        files = [ f for f in files if match(f, SIGCOMP2009_REGEX_Y_GROUP[train]) ==  match(f, 1)]
    cls = [ SIGCOMP2009_CLASS_TAGS[train] + match(f, SIGCOMP2009_REGEX_Y_GROUP[train]) for f in files ]
    return list(zip(files,cls))

# MNIST

def mnist():
    """
    Load MNIST handwritten digits dataset.
    
    Returns:
        torch.utils.data.ConcatDataset: Combined training and test sets of MNIST digits.
    """
    ds = torch.utils.data.ConcatDataset([
        torchvision.datasets.MNIST('./data/mnist/', download=True),
        torchvision.datasets.MNIST('./data/mnist/', download=True, train=False),
    ])
    
    return ds

# Omniglot

OMNIGLOT_PATHS = [ './data/omniglot/images_background', './data/omniglot/images_evaluation' ]

def omniglot(valid=False):
    """
    Load Omniglot character recognition dataset.
     
    Args:
        valid (bool): If False, load background set, otherwise load evaluation set. Defaults to False.
    
    Returns:
        list: List of tuples (file_path, class_label) for each character image.
    """
    files = [f for f in glob.glob(OMNIGLOT_PATHS[valid]+'/**/**/*') if f.endswith('png') or f.endswith('PNG') ]
    cls = [ '-'.join(f.split('/')[-3:-1]) for f in files ]
    return list(zip(files,cls))

# CEDAR

CEDAR_FILE_PATH = './data/cedar'

def cedar():
    """
    Load CEDAR signature dataset.
    
    Returns:
        list: List of tuples (file_path, class_label) for each signature image.
    """
    files = [f for f in glob.glob(CEDAR_FILE_PATH+'/*') if f.endswith('png') or f.endswith('PNG') ]
    cls = [ 'CEDAR-'+f.split('_')[1] for f in files ]
    return list(zip(files,cls))

# Pair datasets

def pds_sigcomp2009():
    """
    Create paired datasets for SIGCOMP 2009 signatures for Siamese network training.
    
    This function creates training and validation pair datasets from the SIGCOMP 2009
    signature dataset, applies appropriate transforms, and displays sample pairs.
    
    Returns:
        tuple: (ds_train, ds_valid, pds_train, pds_valid) containing the raw datasets
               and their corresponding PairDataset wrappers.
    """
    ds_train = sigcomp2009(train=True)
    ds_valid = sigcomp2009(train=False)
    print(f'Sigcomp len(ds_train):{len(ds_train)} len(ds_valid):{len(ds_valid)}')
    tfms = x_transform_signatures()
    pds_train = PairDataset(ds_train, x_transform=tfms, y_transform=y_transform_category(ds_train))
    pds_valid = PairDataset(ds_valid, x_transform=tfms, y_transform=y_transform_category(ds_valid))
    show_pair(pds_valid[0])
    show_pair(pds_valid[1])
    return ds_train, ds_valid, pds_train, pds_valid

def pds_mnist(p=None, validation_digits=(7,2,3)):
    """
    Create paired datasets for MNIST digits for Siamese network training.
    
    This function creates training and validation pair datasets from MNIST,
    separating by specified validation digits. Training set contains all digits
    except validation_digits, while validation set contains only validation_digits.
    
    Args:
        p (float, optional): Proportion of dataset to use (0.0 to 1.0). If None, uses full dataset.
        validation_digits (tuple): Tuple of digit classes (0-9) to use for validation. Defaults to (7,2,3).
    
    Returns:
        tuple: (ds_train, ds_valid, pds_train, pds_valid) containing the raw datasets
               and their corresponding PairDataset wrappers.
    """
    ds = mnist()
    if p is not None:
        ds,_ = torch.utils.data.random_split(ds, [p, 1-p])
    ds_train = torch.utils.data.Subset(ds, [i for i in range(len(ds)) if ds[i][1] not in validation_digits])
    ds_valid = torch.utils.data.Subset(ds, [i for i in range(len(ds)) if ds[i][1] in validation_digits])
    print(f'MNIST {p*100 if p else 100}% len(ds_train):{len(ds_train)} len(ds_valid):{len(ds_valid)}')
    tfms = x_transform_mnist()
    pds_train = PairDataset(ds_train, x_transform=tfms)
    pds_valid = PairDataset(ds_valid, x_transform=tfms, fix_pairing = True)
    show_pair(pds_valid[0])
    show_pair(pds_valid[1])
    return ds_train, ds_valid, pds_train, pds_valid

def pds_cedar():
    """
    Create paired datasets for CEDAR signatures for Siamese network training.
    
    Validation set contains signatures from classes 1-11, training set contains the rest.
    
    Returns:
        tuple: (ds_train, ds_valid, pds_train, pds_valid) containing the raw datasets
               and their corresponding PairDataset wrappers.
    """
    ds = cedar()
    get_y = lambda fp_cls: int(fp_cls[1].split('-')[1])
    ds_train = [(fp, cls) for fp, cls in ds if get_y((fp, cls)) > 11]
    ds_valid = [(fp, cls) for fp, cls in ds if get_y((fp, cls)) <= 11]
    print(f'CEDAR len(ds_train):{len(ds_train)} len(ds_valid):{len(ds_valid)}')
    tfms = x_transform_signatures()
    pds_train = PairDataset(ds_train, x_transform=tfms, y_transform=y_transform_category(ds_train))
    pds_valid = PairDataset(ds_valid, x_transform=tfms, y_transform=y_transform_category(ds_valid))
    show_pair(pds_valid[0])
    show_pair(pds_valid[1])
    return ds_train, ds_valid, pds_train, pds_valid