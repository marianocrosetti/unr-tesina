"""
PyTorch Dataset for generating pairs of images for Siamese network training.

This module provides the PairDataset class which creates matching and non-matching
pairs from a base dataset for training and evaluating Siamese networks.
"""

from src.all_deps import *
from src.tools import *

class PairDataset(Dataset):
    """
    Dataset that generates pairs of samples for Siamese network training.
    
    For each sample in the base dataset, creates two pairs:
    - A matching pair (same class)
    - A non-matching pair (different class)
    
    Args:
        ds: Base dataset with (x, y) pairs where x is data and y is label.
        x_transform: Optional transformation to apply to data. Defaults to None.
        y_transform: Optional transformation to apply to labels. Defaults to None.
        fix_pairing (bool): If True, pre-generates all pairs (deterministic). Defaults to False.
        seed (int): Random seed for pair generation. Defaults to 22.
    """
    def __init__(self, ds, x_transform = None, y_transform = None,fix_pairing = False, seed = 22):
        self.rng = np.random.default_rng(seed)
        self.ds = ds
        self.gds = group_by_y(ds)
        ys = set(self.gds.keys())
        self.y_to_other_y = {y: list(ys.difference({y})) for y in ys}
        self.fix_pairing = []
        if fix_pairing:
            print("Fixing pairs (wait!)...")
            self.fix_pairing = [self.draw(i) for i in tqdm.tqdm(range(len(self)))]
        if x_transform:
            self.x_transform = x_transform
        else:
            self.x_transform = lambda x : x
        if y_transform:
            self.y_transform = y_transform
        else:
            self.y_transform = lambda x : x
        
        
    def __len__(self):
        """Return the length of the dataset (2x base dataset for match/mismatch pairs)."""
        return len(self.ds)*2

    def __getitem__(self, idx):
        """
        Get a pair of samples with their labels.
        
        Args:
            idx (int): Index of the pair to retrieve.
        
        Returns:
            tuple: (x1, x2, y1, y2) where x1, x2 are transformed samples and y1, y2 are their labels.
        """
        x1,x2,y1,y2 = self.get_raw(idx)
        return self.x_transform(x1), self.x_transform(x2), self.y_transform(y1), self.y_transform(y2)
        
    def get_raw(self, idx):
        """
        Get raw (untransformed) pair of samples.
        
        Args:
            idx (int): Index of the pair.
        
        Returns:
            tuple: (x1, x2, y1, y2) raw samples and labels.
        """
        if self.fix_pairing:
            return self.fix_pairing[idx]
        return self.draw(idx)

    def choose(self, xs):
        """
        Randomly choose an element from a list.
        
        Args:
            xs: List of items.
        
        Returns:
            Random item from xs.
        """
        return xs[self.rng.choice(range(len(xs))).item()]
        
    def draw(self, idx):
        """
        Draw a pair of samples (matching or non-matching based on index parity).
        
        Args:
            idx (int): Index determining which sample and whether to create a match/mismatch.
                      Even indices create mismatches, odd indices create matches.
        
        Returns:
            tuple: (x, x_other, y, y_other) pair of samples and their labels.
        """
        x,y = self.ds[idx//2]
        is_match = bool(idx % 2)
        if is_match:
            y_other = y
        else:
            y_other = self.choose(self.y_to_other_y[y])
        x_other = self.choose(self.gds[y_other])
        return x,x_other,y,y_other