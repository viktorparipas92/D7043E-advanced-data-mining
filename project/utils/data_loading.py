from functools import partial
import logging
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    index_with_suffix = f'{idx}{mask_suffix}.*'
    mask_file = list(mask_dir.glob(index_with_suffix))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(
            f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}'
        )


class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        scale: float = 1.0,
        mask_suffix: str = '',
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(file)[0] for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith('.')
        ]
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {images_dir}, '
                f'make sure you put your images there'
            )

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as pool:
            unique = list(
                tqdm(
                    pool.imap(
                        partial(
                            unique_mask_values,
                            mask_dir=self.mask_dir,
                            mask_suffix=self.mask_suffix,
                        ),
                        self.ids,
                    ),
                    total=len(self.ids),
                )
            )

        mask_values = np.unique(np.concatenate(unique), axis=0).tolist()
        self.mask_values = list(sorted(mask_values))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        name_with_suffix = f'{name}{self.mask_suffix}.*'
        name_without_suffix = f'{name}.*'
        mask_file = list(self.mask_dir.glob(name_with_suffix))
        image_file = list(self.images_dir.glob(name_without_suffix))

        assert len(image_file) == 1, \
            f'Either no image or multiple images found for the ID {name}: {image_file}'
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        image = load_image(image_file[0])

        assert image.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {image.size} and {mask.size}'

        image = self.preprocess(
            self.mask_values, image, self.scale, is_mask=False
        )
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

    def preprocess(self, mask_values, pil_image, scale, is_mask):
        new_size = [int(scale * dim) for dim in pil_image.size]
        assert all(dim > 0 for dim in new_size), \
            'Scale is too small, resized images would have no pixel'
        should_resample = Image.NEAREST if is_mask else Image.BICUBIC
        pil_image = pil_image.resize(new_size, resample=should_resample)
        image = np.asarray(pil_image)

        if is_mask:
            return self._mask(image, new_size, mask_values)
        else:
            if image.ndim == 2:
                image = image[np.newaxis, ...]
            else:
                image = image.transpose((2, 0, 1))

            if (image > 1).any():
                image = image / 255.0

            return image

    @staticmethod
    def _mask(image, new_size, mask_values):
        mask = np.zeros(new_size, dtype=np.int64)
        for i, value in enumerate(mask_values):
            if image.ndim == 2:
                mask[image == value] = i
            else:
                mask[(image == value).all(-1)] = i

            return mask


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')