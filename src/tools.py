"""
Utility functions for dataset manipulation, image processing, and PyTorch operations.

This module provides helper functions for grouping datasets, image binarization,
device detection, and various data processing utilities.
"""

from src.all_deps import *

def group_by_y(ds):
    """
    Group dataset samples by their labels.
    
    Args:
        ds: Dataset with (x, y) pairs where y is the label.
    
    Returns:
        dict: Dictionary mapping labels to lists of samples.
    """
    gds = defaultdict(list)
    for x,y in ds:
        gds[y].append(x)
    return gds

def split_ds(ds, is_valid):
    """
    Split dataset into training and validation sets based on a predicate function.
    
    Args:
        ds: Dataset to split.
        is_valid: Function that returns True if a sample belongs to validation set.
    
    Returns:
        tuple: (train, valid) lists containing the split data.
    """
    train,valid = [], []
    for x in ds:
        (train,valid)[is_valid(x)].append(x)
    return train,valid

def binarize_image(np_img):
    """
    Binarize image using Otsu's method with Gaussian blur preprocessing.
    
    Args:
        np_img: Grayscale image as numpy array.
    
    Returns:
        numpy.ndarray: Binarized image (0s and 255s).
    """
    _, otsu_gauss= cv2.threshold(cv2.GaussianBlur(np_img,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu_gauss

def aspect(x):
    """
    Calculate aspect ratio (width/height) of an image.
    
    Args:
        x: Path to image file.
    
    Returns:
        float: Aspect ratio of the image.
    """
    a,b = PIL.Image.open(x).size
    return a/b
    
def ds_mean_aspect(ds):
    """
    Calculate mean aspect ratio across all images in a dataset.
    
    Args:
        ds: Dataset with (x, y) pairs where x is the image path.
    
    Returns:
        float: Mean aspect ratio of all images in the dataset.
    """
    #return [aspect(x) for x,_ in ds]
    return np.mean([aspect(x) for x,_ in ds]).item()

def get_device():
    """
    Get the best available PyTorch device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: Best available device for PyTorch operations.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def get_y_vocab(ds):
    """
    Extract vocabulary mapping for labels in dataset.
    
    Args:
        ds: Dataset with (x, y) pairs where y is the label.
    
    Returns:
        tuple: (itos, stoi) where itos is index-to-string and stoi is string-to-index mapping.
    """
    itos = list(set(y for _,y in ds))
    stoi = {y:i for i,y in enumerate(itos)}
    return itos, stoi

def shapes(l):
    """
    Get shapes of all tensors in a list.
    
    Args:
        l: List of tensors.
    
    Returns:
        list: List of tensor shapes.
    """
    return [x.shape for x in l]

def get_encoder_out_dim(encoder, sample):
    """
    Get output dimensionality of an encoder by running a forward pass with a sample.
    
    Args:
        encoder: Neural network encoder model.
        sample: Sample input tensor.
    
    Returns:
        int: Output dimension of the encoder.
    """
    with torch.no_grad():
        return encoder(sample).shape[-1]

def precision_score_recall_constrained(target, decision_values, required_recall=0.9):
    """
    Find the maximum precision achievable while satisfying a minimum recall constraint.
    
    Uses sklearn's precision_recall_curve to compute all precision-recall pairs,
    then finds the highest precision among those with recall >= required_recall.
    Also returns the threshold that achieves this operating point.
    
    Args:
        target: Ground truth binary labels (0/1).
        decision_values: Predicted scores/probabilities.
        required_recall: Minimum required recall (default 0.9).
    
    Returns:
        tuple: (max_precision, threshold, (precision, recall, thresholds))
            - max_precision: Best precision at or above required_recall.
            - threshold: Decision threshold achieving this precision.
            - Full precision-recall curve data for further analysis.
    """
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(target, decision_values)
    max_precision_by_recall = precision[recall >= required_recall].max()
    t = thresholds[recall[:-1] > required_recall].max()
    return max_precision_by_recall, t, (precision, recall, thresholds)

def precision_recall_accuracy_f1_scores(p, y, th):
    """
    Compute precision, recall, accuracy and F1 score at a given threshold.
    
    Applies threshold to predictions and computes all four classification metrics.
    
    Args:
        p: Predicted scores/probabilities.
        y: Ground truth binary labels.
        th: Decision threshold (predictions >= th are classified as positive).
    
    Returns:
        tuple: (precision, recall, accuracy, f1) scores.
    """
    pred = p >= th
    return precision_score(y, pred), recall_score(y, pred), accuracy_score(y, pred), f1_score(y, pred)
