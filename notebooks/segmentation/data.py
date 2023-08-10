"""Note: The dataset and preprocessing code are largely taken from the StarDist github repository available at https://github.com/stardist"""

import numpy as np

from pathlib import Path
from scipy.ndimage import binary_fill_holes


def get_dsb2018_files(subset):
    assert subset in ["train", "validation", "test"]
    src_dir = Path("../../data/dsb2018/") / subset

    X = sorted(src_dir.rglob('**/images/*.tif'))
    Y = sorted(src_dir.rglob('**/masks/*.tif'))
    assert len(X) > 0
    assert len(X) == len(Y), print(f"X has length {len(X)} and Y has length {len(Y)}")
    assert all(x.name==y.name for x,y in zip(X,Y))

    return X, Y


def get_dsb2018_train_files():
    return get_dsb2018_files(subset="train")


def get_dsb2018_validation_files():
    return get_dsb2018_files(subset="validation")


def get_dsb2018_test_files():
    return get_dsb2018_files(subset="test")


def fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def normalize(img, low, high, eps=1.e-20, clip=True):
    # we have to add a small eps to handle the case where both quantiles are equal
    # to avoid dividing by zero
    scaled = (img - low) / (high - low + eps)

    if clip:
        scaled = np.clip(scaled, 0, 1)

    return scaled


def quantile_normalization(img, quantile_low=0.01, quantile_high=0.998, eps=1.e-20, clip=True):
    """
    First scales the data so that values below quantile_low are smaller
    than 0 and values larger than quantile_high are larger than one.
    Then optionally clips to (0, 1) range.
    """

    qlow = np.quantile(img, quantile_low)
    qhigh = np.quantile(img, quantile_high)

    scaled = normalize(img, low=qlow, high=qhigh, eps=eps, clip=clip)
    return scaled, qlow, qhigh
