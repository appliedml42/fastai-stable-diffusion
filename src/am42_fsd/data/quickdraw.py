from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class QuickDraw(Dataset):
    def __init__(self, base_dir: str):
        """
        :param base_dir: Path where the Numpy Bitmap files are stored.
        https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
        """
        self.base_dir = Path(base_dir)
        self.npy_paths = sorted([x for x in self.base_dir.glob("*npy")])
        self.length = 0

        obj_md = list()
        categories = set()
        for i, npy_path in enumerate(self.npy_paths):
            num_rows, width = np.load(npy_path).shape
            self.length += num_rows
            obj_md.append((self.length, num_rows, width, npy_path))

            category = npy_path.stem
            categories.add(category)

        # Convert to Pandas backed objects for better memory management!
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-1164905242
        self.index = pd.DataFrame(
            obj_md, columns=["length", "num_rows", "width", "npy_path"]
        )
        self.categories = pd.DataFrame(
            [(cat_id, cat) for cat_id, cat in enumerate(sorted(categories))],
            columns=["cat_id", "cat"],
        )

    def __len__(self):
        return self.length

    def __getitem__(self, gi):
        md = self.index.iloc[self.index.length.searchsorted(gi, side="right")]
        offset = md.length - md.num_rows
        li = gi - offset

        with open(md.npy_path.as_posix(), "r") as reader:
            """
            The Numpy Bitmap files are stored as a 2D array of uint8 values. Be careful when using np.memmap!
            It works here because we are reading one specific row at a time and closing the unerlying reader. 
            Still researching if this is a good idea or not.
            """
            image = np.memmap(
                reader, dtype="uint8", mode="r", shape=(md.num_rows, md.width)
            )[li].copy()
            # Need to research if this is the best way to do this!
            #image = torch.asarray(image, copy=True)

        category = md.npy_path.stem
        category_id = self.categories[self.categories.cat == category].cat_id.tolist()[
            0
        ]

        return image, category_id
