"""
Convenience import hook for Jupyter notebooks and interactive Python sessions.

This module aggregates all necessary imports for telegram processing and signature
detection workflows. Intended for use in exploratory data analysis and prototyping.

Simply import this module to have access to:
- All standard libraries (numpy, pandas, cv2, etc.)
- Telegram processing functions (straightening, cropping, QR detection)
- Signature verification models and datasets
- Utility functions (parallel processing, visualization)

Example:
    from src.import_all import *
    # Now you have access to all functions without individual imports
"""

from src.all_deps import *
from src.telegramas_crop_signatures import *
from src.telegram_tools import *
from src.tools import *
from src.datasets import *
from src.models import *
from src.visualization import *
from src.pair_dataset import *


