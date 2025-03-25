# Code modified from PyTorch's dataset utilities: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataset.py
# Date: August 1, 2023


from omegaconf import OmegaConf
import json
from collections import defaultdict
import random
import math
import os
import yaml

from typing import Callable, List
from collections import Counter


from torch.utils.data import Dataset, Subset


class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """

    def __init__(self, dataset: Dataset,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 return_indices: bool = False):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.return_indices = return_indices

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_indices:
            return sample, target, idx
        return sample, target

    def __len__(self):
        return len(self.dataset)


import math
import random
from collections import defaultdict, Counter
from typing import List
from torch.utils.data import Dataset, Subset


def _subsample_balanced(indices_by_class, num_samples):
    """
    Downsample each class to exactly 'num_samples' samples (randomly).
    """
    selected_indices_by_class = {}
    for class_label, idxs in indices_by_class.items():
        if len(idxs) < num_samples:
            raise RuntimeError(
                f"Class {class_label} has only {len(idxs)} samples but asked to downsample to {num_samples}."
            )
        selected_indices_by_class[class_label] = random.sample(idxs, num_samples)
    return selected_indices_by_class


def _upsample_2class_50_50(indices_by_class):
    """
    Forces a 50–50 balance between exactly 2 classes.
      1) Identify minority vs. majority.
      2) Replicate minority up to half of total,
         downsample majority to that same half.

    Returns a list of selected indices for the final, balanced set.
    """
    if len(indices_by_class) != 2:
        raise ValueError("`_upsample_2class_50_50` requires exactly 2 classes.")

    class_labels = list(indices_by_class.keys())
    c0, c1 = class_labels[0], class_labels[1]

    c0_indices = indices_by_class[c0]
    c1_indices = indices_by_class[c1]

    n0 = len(c0_indices)
    n1 = len(c1_indices)
    total = n0 + n1

    # Each class gets half of the final dataset
    target_per_class = total // 2

    # Identify minority vs. majority
    if n0 <= n1:
        minority_indices = c0_indices
        majority_indices = c1_indices
    else:
        minority_indices = c1_indices
        majority_indices = c0_indices

    num_minority = len(minority_indices)
    num_majority = len(majority_indices)

    # Replicate the minority up to target_per_class
    if num_minority < target_per_class:
        times_to_repeat = target_per_class // num_minority
        remainder = target_per_class % num_minority
        final_minority = (minority_indices * times_to_repeat) + \
                         random.sample(minority_indices, remainder)
    else:
        # If minority already bigger than target, random downsample
        final_minority = random.sample(minority_indices, target_per_class)

    # Downsample (or replicate) the majority
    if num_majority > target_per_class:
        final_majority = random.sample(majority_indices, target_per_class)
    else:
        times_to_repeat = target_per_class // num_majority
        remainder = target_per_class % num_majority
        final_majority = (majority_indices * times_to_repeat) + \
                         random.sample(majority_indices, remainder)

    return final_minority + final_majority


def create_subsampled_dataset(
    dataset: Dataset,
    class_ratios: List[float],
    is_test_val: bool = False,
    subsample_balanced: bool = False,
    subsample_upsample: bool = False,
    subsample_balanced_percent_of_total: float = -1.0,
):
    """
    1) For each class, down-sample proportionally based on 'class_ratios'. 
       (Always done unless 'is_test_val' is True.)
    2) If 'subsample_balanced' is True, then after step 1, 
       downsample each class to the same fixed number.
    3) If 'subsample_upsample' is True, then after step 1, 
       do a 50–50 upsample for exactly 2 classes.
    4) If is_test_val, we do a simple balanced subset using the smallest class (often for test/val sets).

    :param dataset: Base dataset (or Subset).
    :param class_ratios: For each class, a ratio for the initial proportional downsampling step.
    :param is_test_val: If True, we always produce a balanced set with max number of samples from the smallest class.
    :param subsample_balanced: If True, after the initial ratio-based downsampling, 
                               we further downsample all classes to the same number.
    :param subsample_upsample: If True, after the initial ratio-based downsampling,
                               we do a 50–50 upsample between 2 classes.
    :param subsample_balanced_percent_of_total: If > 0, fraction of the total dataset for 'subsample_balanced'.
    :return: (subsampled_dataset, label_counts)
    """

    if subsample_balanced and subsample_upsample:
        raise RuntimeError(
            "Cannot both do 'subsample_balanced' and 'subsample_upsample' simultaneously."
        )

    # 1) Gather labels
    if hasattr(dataset, "index"):
        # e.g., a custom dataset with 'index'
        all_labels = dataset.index
    else:
        # If it's a Subset, track back to parent
        all_labels = [
            dataset.dataset.index[dataset.indices[i]] for i in range(len(dataset))
        ]

    # 2) Organize into class -> indices mapping
    indices_by_class = defaultdict(list)
    for i, (class_label, _, _) in enumerate(all_labels):
        indices_by_class[class_label].append(i)

    # 3) If is_test_val, we simply do balanced downsampling across classes, 
    #    each to 'max_samples' = size of smallest class.
    if is_test_val:
        min_samples = min(len(indices) for indices in indices_by_class.values())
        selected_indices = []
        for _, idxs in indices_by_class.items():
            selected_indices.extend(random.sample(idxs, min_samples))

    else:
        # --- Step A: Proportional downsampling first ---
        # We base this on the smallest class size (max_samples).
        # (Adjust as needed if you'd prefer largest class or total dataset as reference.)
        max_samples = min(len(idxs) for idxs in indices_by_class.values())

        #  A.1) For each class, choose 'int(math.floor(max_samples * ratio))'
        #       or keep them all if ratio * max_samples > actual size
        tmp_indices = []  # result of the initial proportional step
        for class_label, ratio in enumerate(class_ratios):
            class_idxs = indices_by_class[class_label]
            desired_count = int(math.floor(max_samples * ratio))
            if desired_count <= len(class_idxs):
                tmp_indices.extend(random.sample(class_idxs, desired_count))
            else:
                # if the ratio * max_samples is bigger than class size,
                # just keep all (no replication)
                tmp_indices.extend(class_idxs)

        # Build a new dictionary from the proportionally downsampled data
        new_indices_by_class = defaultdict(list)
        for idx in tmp_indices:
            lbl = all_labels[idx][0]
            new_indices_by_class[lbl].append(idx)

        # --- Step B: Additional steps if requested ---
        if subsample_balanced:
            # Balanced downsampling to a single 'num_samples' per class
            if subsample_balanced_percent_of_total > 0.0:
                # user-specified fraction
                total_downsample_size = int(subsample_balanced_percent_of_total * len(tmp_indices))
                # each class gets total_downsample_size / #classes
                num_classes = len(new_indices_by_class)
                per_class = max(1, total_downsample_size // num_classes)
            else:
                # or just use the min of the new distribution
                per_class = min(len(idxs) for idxs in new_indices_by_class.values())

            balanced_dict = _subsample_balanced(new_indices_by_class, per_class)
            selected_indices = []
            for _, idxs in balanced_dict.items():
                selected_indices.extend(idxs)

        elif subsample_upsample:
            # 50-50 upsample for exactly 2 classes
            selected_indices = _upsample_2class_50_50(new_indices_by_class)
        else:
            # No further balancing: just use the result of the initial proportion
            selected_indices = tmp_indices

    # Create the final Subset
    subsampled_dataset = Subset(dataset, selected_indices)

    # Print stats
    print("Final subsampled dataset size:", len(subsampled_dataset))
    label_counts = Counter(all_labels[i][0] for i in selected_indices)
    for lbl, count in sorted(label_counts.items()):
        print(f"Label {lbl}: {count} samples")

    return subsampled_dataset, label_counts



def load_normalize_from_file(root, classes: List[str], class_ratios: List[float]):
    # get normalize values for this dataset
    try:
        print(classes)
        data_dir = "/vol/aimspace/users/mildenbd/Projects/ffcv_data" if root is None else root
        if classes is not None:
            path = os.path.join(
                data_dir, f"{'-'.join(classes)}/{'_'.join(map(str, class_ratios))}")
            print(path)
        else:
            path = data_dir
        with open(os.path.join(path, "stats_train.yaml"), 'r') as file:
            stats_train = yaml.safe_load(file)

            normalize = [stats_train['mean'], stats_train['std']]
        return normalize
    except Exception as e:
        print(e)
        print("Could not find mean and std for this data set.")
        print("Using default normalize")

        return None


def create_omegaconf_from_json(json_path: str):
    with open(json_path, 'r') as json_file:
        config_dict = json.load(json_file)
    return OmegaConf.create(config_dict)
