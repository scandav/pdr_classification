from random import seed
from pathlib import Path
from typing import Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import skimage.transform
import torch
from torchvision import transforms
from numpy.typing import ArrayLike
from skimage import io
from torch.utils.data import Dataset


class Resize:
    def __init__(self, size):
        from collections.abc import Iterable
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float64 array
        return skimage.util.img_as_ubyte(resize_image)


aug_transform_pipeline = transforms.Compose([
    Resize(224),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_pipeline = transforms.Compose([
    Resize(224), 
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class DatasetItem(TypedDict):
    images: ArrayLike
    labels: float


class DatasetIR(Dataset):
    """IR Images Dataset."""
    imgs_dir = Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/images")

    def __init__(self, csv_path: Path, transform_pipeline: Optional[transforms.Compose] = None) -> None:

        self.df = pd.read_csv(csv_path)
        self.transform_pipeline = transform_pipeline

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[ArrayLike, float]:

        image_filename = self.df.iloc[idx]['image_uuid'] + ".png"
        image = io.imread(DatasetIR.imgs_dir.joinpath(image_filename)) / 256
        image = np.stack([image, image, image], axis=-1)

        label = self.df.iloc[idx]['proliferative'].astype(np.float32)

        rnd_seed = torch.randint(0, 2 ** 32, size=(1,))[0]
        if self.transform_pipeline:
            seed(rnd_seed)
            image = self.transform_pipeline(image)

        return image, torch.tensor([label]), image_filename

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    pipeline = transforms.Compose([Resize(224)])

    paths = [
        Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/train.csv"),
        Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/validation.csv"),
        Path("/storage/homefs/ds21n601/diabetic_retinopathy_classification/dataset/test.csv")
    ]

    dtypes = ["train", "validation", "test"]

    for path, dtype in zip(paths, dtypes):

        d = DatasetIR(path, transform_pipeline=pipeline)
        dl = torch.utils.data.DataLoader(d, batch_size=8, shuffle=True, num_workers=4)

        idx = 0

        for data in dl:
            imgs, lbls = data

            for img, lbl in zip(imgs, lbls):
                plt.imshow(img)
                plt.title("Proliferative" if lbl == 1.0 else "Non proliferative")
                plt.suptitle(d.df.iloc[idx]['image_uuid'])
                plt.savefig(f'./dataset/inspect_images/{dtype}_{idx:03d}')
                plt.close()
                idx += 1

        print('ciao')
