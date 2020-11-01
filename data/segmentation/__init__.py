import os
import os.path as osp
from PIL import Image

import numpy as np
import torch

__all__ = [ "SegmentDataset" ]

class SegmentDataset:
    CLASSES = [
        'Urban',            # cyan
        'Agriculture',      # yellow
        'Rangeland',        # purple
        'Forest',           # green
        'Water',            # blue
        'Barren',           # white
        'Unkown'            # black
        ]
    COLORS = [
        (0, 255, 255),      # cyan
        (255, 255, 0),      # yellow
        (255, 0, 255),      # purple
        (0, 255, 0),        # green
        (0, 0, 255),        # blue
        (255, 255, 255),    # white
        (0, 0, 0)           # black
        ]

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Image file (x)
        img_files = sorted([ osp.join(root, f)
                            for f in os.listdir(root)
                            if 'sat' in f ])
        # Mask file (y)
        mask_files = sorted([ osp.join(root, f)
                            for f in os.listdir(root)
                            if 'mask' in f ])

        self.pairs = list(zip(img_files, mask_files))

        # Check files are all in order
        for pair in self.pairs:
            img_id = int(osp.basename(pair[0])[:-4].split("_")[0])
            mask_id = int(osp.basename(pair[1])[:-4].split("_")[0])
            assert img_id == mask_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.pairs[idx]

        # Read img (c, h, w)
        img = Image.open(img_file)
        if self.transform is not None:
            img = self.transform(img)

        mask = np.array(Image.open(mask_file)) # (h, w, c)
        target = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for label, c in enumerate(SegmentDataset.COLORS):
            indices = np.where(np.all(mask == np.array(c), axis=-1))[::-1]
            target[indices] = label
        target = torch.LongTensor(target.T)

        return img, target
