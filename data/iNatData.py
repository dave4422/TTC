from collections import Counter
import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets import VisionDataset


class INaturalistNClasses(VisionDataset):

    '''
    Adaption of the pytorch INaturalist dataset implementation
    only containing the data of of the taxonomy 
    subtree under a specific nodes (new classes).

    Can not inherit from the original INaturalist class due to different dir strcuture

    new_classes: list of strings, each string is a category name

    use the val set as test set according to the paper "When Does Contrastive Visual Representation Learning Work?"


    '''

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        classes: List[str] = None,
        #class_ratios: List[float] = None,
    ) -> None:

        self.split = split
        self.classes = classes
        self.store_meta = None

        if self.split not in ["train", "val"]:
            raise RuntimeError(f"Unknown split {split}")

        super().__init__(os.path.join(root, split),
                         transform=transform, target_transform=target_transform)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        # train/07210_Plantae_Tracheophyta_Magnoliopsida_Boraginales_Boraginaceae_Turricula_parryi/58e66757-7d99-455f-b5f4-271ea3b1797f.jpg
        # start at 00000

        # map cat id top full path name
        self.all_categories: List[str] = []
        self.all_categories = sorted(os.listdir(self.root))

        # index of all files: (cls_id, full category id, filepath)
        self.index: List[Tuple[int,int, str]] = []

        if classes is None:
            # full dataset
            print("Using full dataset!")
            for dir_index, dir_name in enumerate(self.all_categories):
                files = os.listdir(os.path.join(self.root, dir_name))
                for fname in files:
                    self.index.append((dir_index, dir_index, fname))

        else:
            # only add samples from cls to index
            for cls_id, cls in enumerate(classes):
                print(classes)
                categories_for_cls = self._get_categories_for_class(cls)
                for cat_id in categories_for_cls:
                    files = os.listdir(os.path.join(
                        self.root, self.all_categories[cat_id]))
                    for fname in files:
                        self.index.append((cls_id, cat_id, fname))

        print(f'Created dataset {self.__class__.__name__} with {len(self)} samples. '\
              f'Split: {self.split}. \n' \
              f'Classes: {self.classes}.'
              )
        cls_counter = Counter(cls_id for (cls_id, _, _) in self.index)
        cls_counts = [cls_counter.get(i, 0) for i in range(len(self.classes))]
        print(f'Class counts: {cls_counts}')

        


    def _get_categories_for_class(self, cls: str) -> List[int]:
        '''
        Returns all categorie ids that are in the subtree of the given class
        '''
        cats: List[int] = []
        for dir_name in self.all_categories:
            # remove the numeric prefix
            cat_name = dir_name.split("_", 1)[1].lower()

            if cat_name.startswith(cls.lower()):
                cats.append(int(dir_name.split("_", 1)[0]))

        return cats

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        class_id, cat_id, fname = self.index[index]
        img = Image.open(os.path.join(
            self.root, self.all_categories[cat_id], fname))

        # target: Any = []

        # target.append(cat_id)

        target = class_id  # int number for the class

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



    def __len__(self) -> int:
        return len(self.index)

    def _check_integrity(self) -> bool:
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0
