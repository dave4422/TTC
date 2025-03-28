import math
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor


# if _SKLEARN_AVAILABLE:
#     from sklearn.utils import shuffle as sk_shuffle
# else:  # pragma: no cover
#     warn_missing_pkg("sklearn", pypi_name="scikit-learn")


class Identity(torch.nn.Module):
    """An identity class to replace arbitrary layers in pretrained models.

    Example::

        from pl_bolts.utils import Identity

        model = resnet18()
        model.fc = Identity()

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


# @under_review()
# def balance_classes(
#     X: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray, Sequence[int]], batch_size: int  # noqa: N803
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Makes sure each batch has an equal amount of data from each class. Perfect balance.

#     Args:
#         X: input features
#         y: mixed labels (ints)
#         batch_size: the ultimate batch size

#     """
#     if not _SKLEARN_AVAILABLE:  # pragma: no cover
#         raise ModuleNotFoundError("You want to use `shuffle` function from `scikit-learn` which is not installed yet.")

#     num_classes = len(set(y))
#     num_batches = math.ceil(len(y) / batch_size)
#     # sort by classes
#     final_batches_x: List[list] = [[] for i in range(num_batches)]
#     final_batches_y: List[list] = [[] for i in range(num_batches)]

#     # Y needs to be np arr
#     y = np.asarray(y)

#     # pick chunk size for each class using the largest split
#     chunk_sizes = []
#     for class_i in range(num_classes):
#         mask = class_i == y
#         y_sub = y[mask]
#         chunk_sizes.append(math.ceil(len(y_sub) / num_batches))
#     chunk_size = max(chunk_sizes)
#     # force chunk size to be even
#     if chunk_size % 2 != 0:
#         chunk_size -= 1

#     # divide each class into each batch
#     for class_i in range(num_classes):
#         mask = class_i == y
#         x_sub = X[mask]
#         y_sub = y[mask]

#         # shuffle items in the class
#         x_sub, y_sub = sk_shuffle(x_sub, y_sub, random_state=123)

#         # divide the class into the batches
#         for i_start in range(0, len(y_sub), chunk_size):
#             batch_i = i_start // chunk_size
#             i_end = i_start + chunk_size

#             if len(final_batches_x) > batch_i:
#                 final_batches_x[batch_i].append(x_sub[i_start:i_end])
#                 final_batches_y[batch_i].append(y_sub[i_start:i_end])

#     # merge into full dataset
#     final_batches_x = np.concatenate([np.concatenate(x, axis=0) for x in final_batches_x if len(x) > 0], axis=0)
#     final_batches_y = np.concatenate([np.concatenate(x, axis=0) for x in final_batches_y if len(x) > 0], axis=0)
#     return final_batches_x, final_batches_y
