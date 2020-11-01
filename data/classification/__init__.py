import os
import os.path as osp
from PIL import Image

__all__ = [ "ImageDataset" ]

class ImageDataset:

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.files = [ osp.join(root, f) for f in os.listdir(root) ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        # Read img file
        img = Image.open(f)
        img = self.transform(img)
        # Parse class of img
        label = int(osp.basename(f)[:-4].split("_")[0])

        return img, label
