import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import lmdb
import pickle
import io
# Updated LMDBDataset class
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, processor, max_tgt_length=128, height=96, width=96, do_aug=True):
        self.lmdb_path = lmdb_path
        self.processor = processor
        self.max_target_length = max_tgt_length
        self.height = height
        self.width = width
        self.do_aug = do_aug

        # Open the LMDB environment
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        # Get total number of entries
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Convert index to key
        # key = f'{idx:07}'.encode('ascii')
        key = f'{idx:010}'.encode('ascii')

        # Open a transaction and fetch data
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
            if data is None:
                raise IndexError(f"Index {idx} is out of range.")
            data = pickle.loads(data)
            img_bytes = data['image']
            text = data['text']

        # Convert bytes to image
        img_buffer = io.BytesIO(img_bytes)
        image = Image.open(img_buffer).convert("L")

        # Process image
        pixel_values = self.processor.process_image(
            image,
            size_dict=(self.height, self.width),
            rescale_factor=0.00392156862745098,
            image_mean=0.5,
            image_std=0.5,
            do_aug=self.do_aug,
            do_resize=True,
            do_rescale=True,
            do_normalize=True
        )

        # Tokenize text
        labels = self.processor.tokenize(text, self.max_target_length)
        # Ensure PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.PAD_ID else -100 for label in labels]

        encoding = {"pixel_values": pixel_values, "labels": torch.tensor(labels)}
        return encoding

    def __getstate__(self):
        # Remove the LMDB environment before pickling
        state = self.__dict__.copy()
        del state['env']
        return state

    def __setstate__(self, state):
        # Restore the LMDB environment after unpickling
        self.__dict__.update(state)
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )