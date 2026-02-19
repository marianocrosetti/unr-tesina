"""
All external dependencies for the signature verification project.

This module consolidates all external library imports so that other modules
can simply use 'from all_deps import *' to access all required dependencies.

Includes dependencies for:
- PyTorch/ML workflows (signature verification)
- Image processing and computer vision (OpenCV, PIL, scikit-image)
- Telegram processing (QR detection, PDF conversion, OCR)
- Scientific computing and data analysis
"""

from pathlib import Path
from pypdf import PdfReader

# Standard library
import glob
import re
import os
import sys
import gc
import time
import copy
import tempfile
import functools
import shutil
from collections import defaultdict
from math import *

# Image processing
import cv2
import PIL
from PIL import Image, ImageDraw, ImageOps
import imageio
from shutil import copyfile

# Computer vision and QR detection
from pyzbar.pyzbar import decode
import pytesseract

# PDF processing
from pdf2image import convert_from_path, convert_from_bytes, pdfinfo_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

# Data manipulation
import numpy as np
import pandas as pd

# Scientific computing
from scipy import ndimage, spatial, stats
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops, regionprops_table

# Machine Learning
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, Normalizer

# PyTorch
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset

# Visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, precision_score, recall_score, accuracy_score, f1_score

# Progress bars and parallel processing
import tqdm
import joblib
import multiprocessing
from multiprocessing import Pool

# Serialization
import pickle5 as pickle

# Optional: fastbook utilities (comment out if not available)
try:
    from fastbook import *
except ImportError:
    pass  # fastbook is optional

