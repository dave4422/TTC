
from medmnist import PneumoniaMNIST, BreastMNIST, ChestMNIST, OCTMNIST
from collections import Counter
from matplotlib import use
import torch
from torch.utils.data import DataLoader, random_split

from lightning import LightningDataModule
from typing import List, Optional

import lightning as pl
from torch.utils.data import Dataset

import glob
import json
from typing import Optional

from sklearn.model_selection import train_test_split

# Simplified dataset imports

from data.iNatData import INaturalistNClasses
from data.utils import create_subsampled_dataset, ApplyTransform
from torch.utils.data import WeightedRandomSampler
from data.cardiac import CardiacData

from torch.utils.data import Dataset
from tqdm import tqdm

from torchvision.datasets import ImageFolder
from PIL import Image
import os

import numpy as np


class InMemoryDataset(Dataset):
    def __init__(self, dataset, transform=None, load_in_mem=True, is_chest=False):
        self.transform = transform
        self.dataset = dataset
        self.data = None
        self.new_labels = []

        # if is_chest:

        print("InMemoryDataset")
        labels_count = [0, 0]
        if load_in_mem:
            if self.data == None:
                self.data = []
                for idx in tqdm(range(len(self.dataset)), desc="Loading data into memory"):
                    # print(self.dataset[idx])
                    x = self.dataset[idx]
                    _, label = x
                    labels_count[int(label)] += 1
                    self.data.append(x)
            self.labels_count = labels_count
            print("labels count", self.labels_count)
        else:
            self.data = dataset

    def get_cls_num_list(self):
        return self.labels_count

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # print("here")
        img, label = self.data[idx]

        if self.new_labels != []:
            new_labels = self.new_labels[idx]
        else:
            new_labels = -1

        if self.transform:
            img = self.transform(img)
        return img, label, new_labels


class AggregatedLabels(Dataset):
    def __init__(self, dataset, transform=None, load_in_mem=True, is_chest=False):
        self.transform = transform
        self.dataset = dataset
        self.data = None
        self.new_labels = []

        labels_count = [0, 0]

        if self.data == None:
            self.data = []
            for idx in tqdm(range(len(self.dataset)), desc="Loading data into memory"):
                # print(self.dataset[idx])
                x = self.dataset[idx]
                img, label = x

                if label != 0 or label != 3:
                    continue
                new_label = 0 if label == 0 else 1
                labels_count[int(new_label)] += 1

                x = (img, new_label)
                self.data.append(x)
        self.labels_count = labels_count

        print("labels count", self.labels_count)

    def get_cls_num_list(self):
        return self.labels_count

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # print("here")
        img, label = self.data[idx]

        if self.transform:
            img = self.transform(img)
        return img, label


class SingleClassImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, label=0):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            label (int): Label to assign to all images.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.label = label
        self.image_files = [
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.label


class SimpleImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform=None,
        val_transform=None,
        persistent_workers=True,
        in_memory=False
    ):
        """
        Args:
            train_dir (str): Path to the training images directory.
            val_dir (str): Path to the validation images directory.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            image_size (int): Size to which images will be resized.
            transform (transforms.Compose, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.in_memory = in_memory

        self.transform = train_transform
        self.test_transform = val_transform

    def setup(self, stage: Optional[str] = None):
        """Load datasets. This method is called by Lightning with the 'fit' and 'test' stages."""
        if self.transform is None:
            raise ValueError("You must provide a transform.")

        self.train_dataset = ImageFolder(
            self.train_dir,
            transform=self.transform

        )
        self.val_dataset = ImageFolder(
            self.val_dir,
            transform=self.transform,

        )

        self.test_dataset = ImageFolder(
            self.test_dir,
            transform=self.test_transform,
        )
        if self.in_memory:
            self.train_dataset = InMemoryDataset(self.train_dataset)
            self.val_dataset = InMemoryDataset(self.val_dataset)
            self.test_dataset = InMemoryDataset(self.test_dataset)
        # self.train_dataset = InMemoryDataset(self.train_dataset)
        # self.val_dataset = InMemoryDataset(self.val_dataset)
        # self.test_dataset = InMemoryDataset(self.test_dataset)

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,  # Assuming using validation set for testing; modify as needed
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class NPYLiverDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list): List of paths to .npy files.
            labels (dict): Dictionary mapping participant ID to labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        participant_id = int(os.path.splitext(os.path.basename(file_path))[0])
        # Get label, handle missing labels if needed
        label = self.labels.get(participant_id, None)

        if label is None:
            raise ValueError(
                f"Label not found for participant ID: {participant_id}")

        try:
            image_3d = np.load(file_path)
            slice_2d = image_3d[:360, :, 14]  # Extract slice and crop

            # Normalize to 0-1 (or 0-255 and then to tensor if transforms expect 0-1)
            min_val = np.min(slice_2d)
            max_val = np.max(slice_2d)
            if max_val > min_val:
                normalized_slice = (slice_2d - min_val) / (max_val - min_val)
            else:
                # or np.ones_like * 0.5 for gray mid-value
                normalized_slice = np.zeros_like(slice_2d)

            image = np.expand_dims(normalized_slice, axis=0).astype(
                np.float32)  # Add channel dimension, ensure float32 for PyTorch

            if self.transform:
                # Apply transforms after loading and slicing
                image = self.transform(image)

        except Exception as e:
            print(f"Error loading or processing file: {file_path}, Error: {e}")
            raise

        return image, label


class NPYLiverDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,  # Base directory containing .npy files
        label_file: str,  # Path to the liver_diagnosis_dict.json
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        val_size: float = 0.1,
        test_size: float = 0.2,
        random_seed: int = 42,
        persistent_workers=True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.label_file = label_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.val_size = val_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.persistent_workers = persistent_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # No data download or preparation needed in this case as data is assumed to be in place.
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            # Load labels
            with open(self.label_file, 'r') as f:
                labels = json.load(f)
                
            labels = {int(k): v for k, v in labels.items()}

            # Get list of .npy files
            np_files = glob.glob(os.path.join(self.data_dir, "*.npy"))
            all_ids = [int(os.path.splitext(os.path.basename(file))[0]) for file in np_files] 
            
            labeled_ids = set(labels.keys())
            valid_ids = list(set(all_ids) & labeled_ids)
            
            valid_ids_with_slices = []
            filtered_out_count = 0

            for participant_id in all_ids:
                if participant_id in labeled_ids:
                    file_path = os.path.join(self.data_dir, f"{participant_id}.npy")
                    try:
                        image_3d = np.load(file_path)
                        z_size = image_3d.shape[2]
                        if z_size >= 15:
                            valid_ids_with_slices.append(participant_id)
                        else:
                            filtered_out_count += 1
                    except Exception as e:
                        print(f"Error loading file {file_path} during slice check: {e}")
                        filtered_out_count += 1 # Consider it filtered out if loading fails during check
                else:
                    filtered_out_count += 1 # Filter out if no label

            valid_ids = valid_ids_with_slices # Rename for clarity in rest of the code
            
            ids = valid_ids

            # Split IDs into train, val, test sets
            train_ids, temp_ids = train_test_split(
                ids, test_size=self.val_size + self.test_size, random_state=self.random_seed)
            val_ids, test_ids = train_test_split(temp_ids, test_size=self.test_size / (self.val_size + self.test_size) if (
                self.val_size + self.test_size) > 0 else 0.5, random_state=self.random_seed)  # Handle case where val_size + test_size is 0

            train_files = [os.path.join(
                self.data_dir, f"{id}.npy") for id in train_ids]
            val_files = [os.path.join(
                self.data_dir, f"{id}.npy") for id in val_ids]
            test_files = [os.path.join(
                self.data_dir, f"{id}.npy") for id in test_ids]

            train_labels_dict = {id: labels[id]
                                 for id in train_ids if id in labels}
            val_labels_dict = {id: labels[id]
                               for id in val_ids if id in labels}
            test_labels_dict = {id: labels[id]
                                for id in test_ids if id in labels}

            self.train_dataset = NPYLiverDataset(
                train_files, train_labels_dict, transform=self.train_transform)
            self.val_dataset = NPYLiverDataset(
                val_files, val_labels_dict, transform=self.val_transform)
            self.test_dataset = NPYLiverDataset(
                test_files, test_labels_dict, transform=self.test_transform)

        if stage == 'test':
            if self.test_dataset is None:
                with open(self.label_file, 'r') as f:
                    labels = json.load(f)
                labels = {int(k): v for k, v in labels.items()}

                np_files = glob.glob(os.path.join(self.data_dir, "*.npy"))
                all_ids = [int(os.path.splitext(os.path.basename(file))[0]) for file in np_files]

                labeled_ids = set(labels.keys())
                valid_ids_with_slices = []

                for participant_id in all_ids:
                    if participant_id in labeled_ids:
                        file_path = os.path.join(self.data_dir, f"{participant_id}.npy")
                        try:
                            image_3d = np.load(file_path)
                            z_size = image_3d.shape[2]
                            if z_size >= 15:
                                valid_ids_with_slices.append(participant_id)
                        except Exception as e:
                            pass # Silently skip if loading fails during check
                valid_ids = valid_ids_with_slices
                
                ids = valid_ids

                # Split IDs (using the same split as in 'fit' for consistency if needed, or load a pre-defined split)
                _, temp_ids = train_test_split(
                    ids, test_size=self.val_size + self.test_size, random_state=self.random_seed)
                _, test_ids = train_test_split(temp_ids, test_size=self.test_size / (self.val_size + self.test_size) if (
                    self.val_size + self.test_size) > 0 else 0.5, random_state=self.random_seed)

                test_files = [os.path.join(
                    self.data_dir, f"{id}.npy") for id in test_ids]
                test_labels_dict = {id: labels[id]
                                    for id in test_ids if id in labels}
                self.test_dataset = NPYLiverDataset(
                    test_files, test_labels_dict, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class NClassesDataModule(LightningDataModule):
    def __init__(self, data_dir: str, classes: List[str],
                 train_transform=None,
                 val_transform=None,
                 num_workers: int = 32,
                 data_set: str = "inat21",
                 class_ratios: Optional[List[float]] = None,
                 batch_size: int = 64,
                 seed: int = 42,
                 subsample_balanced: bool = False,
                 subsample_upsample: bool = False,
                 drop_last: bool = False,
                 weighte_sampling: bool = False,
                 shuffle: bool = True,
                 pin_memory=True,
                 persistent_workers=True,
                 in_memory=False):

        super().__init__()
        self.data_set = data_set
        self.root_dir = data_dir
        self.classes = classes
        self.class_ratios = class_ratios or []
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.weighte_sampling = weighte_sampling
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.name = f"{self.data_set}_{len(self.classes)}_classes"
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.subsample_balanced = subsample_balanced
        self.subsample_upsample = subsample_upsample
        self.persistent_workers = persistent_workers
        self.in_memory = in_memory

        if abs(sum(self.class_ratios) - 1.0) > 1e-5:
            raise RuntimeError("Class ratios must sum to 1.0")

        if subsample_balanced and subsample_upsample:
            raise RuntimeError(
                "Cannot downsample and subsample to count min, then upsample - they are mutually exclusive")

    def setup(self, stage=None):
        if self.data_set == "inat21":
            self.setup_inat_dataset()
        elif self.data_set == "imagenet":
            raise NotImplementedError(
                "Imagenet dataset setup not implemented yet.")
        else:
            raise ValueError(f"Unknown dataset {self.data_set}")

    def setup_inat_dataset(self):
        # Shared setup logic for iNaturalist dataset
        generator2 = torch.Generator().manual_seed(self.seed)
        test_dataset = INaturalistNClasses(
            self.root_dir, split="val", transform=self.val_transform, classes=self.classes)
        self.test_dataset, _ = create_subsampled_dataset(
            test_dataset, None, is_test_val=True)

        total_dataset = INaturalistNClasses(
            self.root_dir, split="train", classes=self.classes)
        train_dataset, val_dataset = random_split(
            total_dataset, [0.95, 0.05], generator=generator2)

        print(f"train dataset size: {len(train_dataset)}")
        print(f"val dataset size: {len(val_dataset)}")
        print(f"test dataset size: {len(self.test_dataset)}")

        train_subsampled, train_counts = create_subsampled_dataset(
            train_dataset, self.class_ratios, is_test_val=False,
            subsample_upsample=self.subsample_upsample,
            subsample_balanced=self.subsample_balanced)
        val_subsampled, val_counts = create_subsampled_dataset(
            val_dataset, None, is_test_val=True)

        # Wrap datasets with InMemoryDataset to preload data into RAM
        print(f"self.in_memory: {self.in_memory}")
        if self.in_memory:
            train_subsampled = InMemoryDataset(
                train_subsampled, transform=None)
            val_subsampled = InMemoryDataset(val_subsampled, transform=None)
            test_dataset = InMemoryDataset(self.test_dataset, transform=None)
            print("Data loaded into memory")

        if self.train_transform is not None:

            self.train_dataset = ApplyTransform(
                train_subsampled, self.train_transform)
            self.val_dataset = ApplyTransform(
                val_subsampled, self.val_transform)
            # self.test_dataset = ApplyTransform(
            #     test_dataset, self.val_transform)
        else:
            self.train_dataset = train_subsampled
            self.val_dataset = val_subsampled
            self.test_dataset = test_dataset

        self.train_counts = [train_counts[cls]
                             for cls in range(len(self.classes))]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            # prefetch_factor=2,
        )

    def val_dataloader(self):
        # shuffle required to have both classes in each batch for sim calc
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class DynamicWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, weights, num_samples, replacement=False):
        super().__init__(weights, num_samples, replacement)

    def update_weights(self, new_weights):
        self.weights = torch.as_tensor(
            new_weights, dtype=torch.double, device='cpu')
        self.num_samples = len(new_weights)


class NClassesDataModuleRandomSampling(NClassesDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = None  # initialize sampler in setup method

    def setup(self, stage=None):
        super().setup(stage)
        initial_weights = self.get_initial_weights()
        self.sampler = DynamicWeightedRandomSampler(
            initial_weights, len(self.train_dataset))

    def get_initial_weights(self):
        return torch.ones(len(self.train_dataset))

    def update_sampler_weights(self, new_weights):
        self.sampler.update_weights(new_weights)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True, shuffle=False,  # Set shuffle to False when using a custom sampler
            persistent_workers=True,
            num_workers=self.num_workers, drop_last=self.drop_last,
            sampler=self.sampler
        )


class CardiacDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 train_transform=None,
                 val_transform=None,
                 num_workers: int = 32,
                 minority_class: str = "cad_broad",
                 batch_size: int = 64,
                 seed: int = 42,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 pin_memory=True,
                 use_pil=False,
                 subsample_balanced_train=False):

        super().__init__()
        self.minority_class = minority_class
        self.root_dir = data_dir

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.subsample_balanced_train = subsample_balanced_train
        self.use_pil = use_pil

        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):

        self.test_dataset = CardiacData(img_root=self.root_dir, split="test", transform = self.val_transform, minority_class = self.minority_class, use_pil = self.use_pil)     
        train_dataset = CardiacData(img_root=self.root_dir, split="train", minority_class = self.minority_class, use_pil = self.use_pil)
        val_dataset = CardiacData(img_root=self.root_dir, split="val",  minority_class = self.minority_class, use_pil = self.use_pil)


        print("train dataset size: ", len(train_dataset))
        print("val dataset size: ", len(val_dataset))
        print("test dataset size: ", len(self.test_dataset))

        if self.subsample_balanced_train:
            train_dataset, train_counts = create_subsampled_dataset(
                train_dataset, None, subsample_balanced=True, subsample_balanced_percent_of_total=0.05)
            self.classes = [0, 1]

        print("applying transforms")
        # apply transforms here for backward compatibility reasons (ApplyTransform also returns the index)
        self.train_dataset = ApplyTransform(
            train_dataset, self.train_transform, )
        self.val_dataset = ApplyTransform(val_dataset, self.val_transform,)

        if not self.subsample_balanced_train:
            labels_at_index_0 = [tup[0] for tup in train_dataset.index]
            train_counts = Counter(labels_at_index_0)
            self.classes = list(set(labels_at_index_0))
        self.train_counts = [train_counts[cls] for cls in self.classes]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory, shuffle=self.shuffle, persistent_workers=True,
            num_workers=self.num_workers, drop_last=self.drop_last,
        )

    def val_dataloader(self):
        # shuffle required to have both classes in each batch for sim calc
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class MedMNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str,
                 train_transform=None,
                 val_transform=None,
                 num_workers: int = 32,
                 data_set: str = "",

                 batch_size: int = 64,
                 seed: int = 42,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 pin_memory=True):

        super().__init__()
        self.data_set = data_set
        self.root_dir = data_dir
        print(batch_size)

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        dataset_mapping = {
            "pneumonia": PneumoniaMNIST,
            "breast": BreastMNIST,
            "chest": ChestMNIST,
            "oct": OCTMNIST
        }

        if self.data_set in dataset_mapping:
            self.setup_dataset(dataset_mapping[self.data_set])
        else:
            raise ValueError(f"Unknown dataset {self.data_set}")

    def setup_dataset(self, dataset_cls):
        self.test_dataset = self.create_dataset(dataset_cls, 'test')
        self.train_dataset = self.create_dataset(dataset_cls, 'train')
        self.val_dataset = self.create_dataset(dataset_cls, 'val')

        if dataset_cls == ChestMNIST or dataset_cls == OCTMNIST:
            self.test_dataset = AggregatedLabels(self.test_dataset)
            self.train_dataset = AggregatedLabels(self.train_dataset)
            self.val_dataset = AggregatedLabels(self.val_dataset)

        labels_flattened = self.train_dataset.labels.flatten()
        train_counts = Counter(labels_flattened)
        self.classes = list(set(labels_flattened))
        self.train_counts = [train_counts[cls] for cls in self.classes]

        print(train_counts)
        print(f"train dataset size: {len(self.train_dataset)}")
        print(f"val dataset size: {len(self.val_dataset)}")
        print(f"test dataset size: {len(self.test_dataset)}")

    def create_dataset(self, dataset_cls, split):

        if split == 'train':
            return dataset_cls(
                split=split,
                transform=self.train_transform,
                target_transform=None,
                download=True,
                as_rgb=True,
                root=self.root_dir,
                size=224,
            )
        return dataset_cls(
            split=split,
            transform=self.val_transform,
            target_transform=None,
            download=True,
            as_rgb=True,
            root=self.root_dir,
            size=224,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory, shuffle=self.shuffle, persistent_workers=True,
            num_workers=self.num_workers, drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
