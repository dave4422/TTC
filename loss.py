import torch.distributed.nn.functional as dist_f
from pytorch_lightning import LightningModule
import torch.distributed as dist
from ast import Not
from typing import List, final
from sympy import use
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn


from lightning import LightningModule


class AlignUniformLoss(LightningModule):
    '''
    url:(https://arxiv.org/pdf/2005.10242.pdf)
    '''

    def __init__(self,
                 lam: float = 1.0,
                 lam_align: float = 1.0,
                 alpha: float = 2.0,
                 t: float = 2.0,

                 ) -> None:
        super().__init__()

        # lam : hyperparameter balancing the two losses
        self.lam = lam
        self.alpha = alpha
        self.t = t
        self.lam_align = lam_align

    def lalign(self, x, y, ):
        return (x - y).norm(dim=1).pow(self.alpha).mean()

    def lunif(self, x):
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-self.t).exp().mean().log()

    def forward(self,
                features,
                # const_numerator = False
                ) -> torch.Tensor:
        '''

        # bsz : batch size (number of positive pairs)
        # d : latent dim
        # x : Tensor, shape=[bsz, d]
        # latents for one side of positive pairs
        # y : Tensor, shape=[bsz, d]
        # latents for the other side of positive pairs

        '''

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        bsz = features.shape[0]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        x = contrast_feature[:bsz]
        y = contrast_feature[bsz:]

        loss = self.lam_align * \
            self.lalign(x, y) + self.lam * (self.lunif(x) + self.lunif(y)) / 2

        return loss


class SupConLoss(LightningModule):
    '''
    Supervise contrstive loss function

    temperature: Temperature at which softmax evaluation is done. Temperature
            must be a python scalar or scalar Tensor of numeric dtype.

    contrast_mode:  All the views of all samples are used as anchors 

    base_temperature:

    denominator_mode: LossDenominatorMode specifying which positives to include
        in contrastive denominator. See documentation above for more details.
    positives_cap: Integer maximum number of positives *other* than
        augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
        Including augmentations, a maximum of (positives_cap + num_views - 1)
        positives is possible. This parameter modifies the contrastive numerator
        by selecting which positives are present in the summation, and which
        positives contribure to the denominator if denominator_mode ==
        enums.LossDenominatorMode.ALL.


    '''

    def __init__(self,
                 temperature: float = 0.5,
                 contrast_mode: str = "ALL_VIEWS",
                 base_temperature: float = 0.07,
                 min_class: int = None,
                 ratio_remove_majority_class_from_numerator: float = -1.0,
                 ratio_remove_majority_class_from_denominator=-1.0,
                 weighting_positives=-1.0,
                 reweight_global_min_loss=-1.0,

                 ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.majority_class = 0 if int(min_class) == 1 else 1
        self.minority_class = int(min_class)
        self.ratio_remove_majority_class_from_numerator = ratio_remove_majority_class_from_numerator

        self.weighting_positives = weighting_positives

        self.ratio_remove_majority_class_from_denominator = ratio_remove_majority_class_from_denominator

        self.reweight_global_min_loss = reweight_global_min_loss

    def forward(self,
                features,
                labels=None,
                ) -> torch.Tensor:
        '''

        Code is inspired by tensor flow code of the orginal authors

        features: A tensor with a min of 3 dime, structured as [bsz, n_views, ...], where:
                - bsz (int): Batch size, representing the number of samples in a batch.
                - n_views (int): Number of views or augmentations per sample.

        labels: 1-dimensional tensor of binary labels, either 0 or 1, with length batch_size:
                - batch_size (int): Number of samples in the batch.

        '''

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if not isinstance(labels, torch.Tensor) and not labels is None:
            labels = torch.tensor(labels, dtype=torch.float32)
            # print("here")

        calc_loss_maj_min_sep = False
        mask_unsup = None
        mask_maj = None
        mask_min = None

        ratio = self.ratio_remove_majority_class_from_numerator

        mask_unsup = torch.eye(batch_size, dtype=torch.float32).to(self.device)

        if labels is None:
            # SimCLR unsupervised loss
            mask = mask_unsup
        elif self.ratio_remove_majority_class_from_numerator > 0.0:
            labels = labels.contiguous().view(-1, 1) if labels is not None else None

            # Ensure labels dimensions match expected shapes if labels are not None
            if labels is not None and labels.shape[0] != batch_size:
                raise ValueError(
                    'Number of labels does not match number of features')

            # SupCon with fewer positives in majority class and no reweighting
            numerator_mask, ratio = self._create_numerator_mask(labels,
                                                                self.ratio_remove_majority_class_from_numerator,
                                                                self.minority_class, self.majority_class)
            mask = numerator_mask
        else:
            # Create masks for supervised loss with handling for majority and minority class balance
            min_indices = labels == self.minority_class
            maj_indices = labels == self.majority_class

            labels = labels.contiguous().view(-1, 1) if labels is not None else None

            # Ensure labels dimensions match expected shapes if labels are not None
            if labels is not None and labels.shape[0] != batch_size:
                raise ValueError(
                    'Number of labels does not match number of features')

            mask = torch.eq(labels, labels.T).float().to(self.device)
            mask_maj, mask_min = mask.clone(), mask.clone()

            mask_maj[min_indices] = 0.0
            mask_maj[:, min_indices] = 0.0
            mask_min[maj_indices] = 0.0
            mask_min[:, maj_indices] = 0.0

        contrast_count = features.shape[1]  # num of augmentations
        # stack all the augmentations on top of each other
        # first features 0 to bsz and then their augmentations
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'ALL_VIEWS':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        # Generate `logits`, the tensor of (temperature-scaled) dot products of the
        # anchor features with all features.

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        if mask_maj is not None:
            mask_maj = mask_maj.repeat(anchor_count, contrast_count)

        if mask_min is not None:
            mask_min = mask_min.repeat(anchor_count, contrast_count)
        if mask_unsup is not None:
            mask_unsup = mask_unsup.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        # logits_mask is used to get the denominator(positives and negatives)
        logits_mask = torch.scatter(
            # tensor filled with the scalar value 1, with the same size as mask
            torch.ones_like(mask),
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1,
                                                         1).to(self.device),  # index
            0  # fill value
        )

        mask = mask * logits_mask
        if mask_unsup is not None:
            mask_unsup = mask_unsup * logits_mask

        if mask_min is not None:
            mask_maj = mask_maj * logits_mask
            mask_min = mask_min * logits_mask

        # mask out some of the majority class logits from the denominator
        # not used in paper
        if self.ratio_remove_majority_class_from_denominator > 0.0:
            logits_mask = self._remove_majority_class_from_denominator_mask(
                logits_mask, labels, anchor_count, contrast_count)
        # compute log_prob
        # all samples except self contrast case
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / \
            (mask.sum(1))  # num stability

        if self.base_temperature > 0.0:
            loss = - self.temperature/self.base_temperature * mean_log_prob_pos
        else:
            loss = - 1.0 * mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()

        labels = torch.arange(batch_size, device=self.device, dtype=torch.long)
        labels = labels+batch_size-1  # Remove sim to self
        labels = torch.cat([labels,  torch.arange(
            batch_size, device=self.device, dtype=torch.long)], dim=0)

        # Logits with self sim removed
        clean_logits = exp_logits[~torch.eye(
            batch_size*anchor_count).bool()].view(batch_size*anchor_count, -1)
        ####

        return loss, clean_logits, labels

    def _remove_majority_class_from_denominator_mask(self, logits_mask, labels, anchor_count, contrast_count):
        ratio = self.ratio_remove_majority_class_from_denominator  # ratio of maj to remove
        majority_indices = torch.where(labels == self.majority_class)[0]

        # select ratio number of indices from the majority class at random
        selected_majority_indices = torch.randperm(majority_indices.size(0))[
            :int(ratio * majority_indices.size(0))]
        selected_majority_indices = majority_indices[selected_majority_indices]
        mask = torch.ones((labels.size(0), labels.size(0)),
                          device=labels.device)

        mask[:, selected_majority_indices] = 0
        mask = mask.repeat(anchor_count, contrast_count)

        # Move the tensor back to the original device.
        return logits_mask * mask

    def _create_numerator_mask(self, labels, ratio, minority_class, majority_class, warm_up=15):

        ratio = 1.0 - ratio
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"Ratio must be between 0 and 1, but is {ratio}")

        minority_indices = torch.where(labels == minority_class)[0]
        majority_indices = torch.where(labels == majority_class)[0]

        selected_majority_indices = torch.randperm(majority_indices.size(0))[
            :int(ratio * majority_indices.size(0))]

        selected_majority_indices = majority_indices[selected_majority_indices]
        combined_indices = torch.cat(
            [minority_indices, selected_majority_indices])

        selected_labels = labels[combined_indices].view(-1, 1)
        mask_partial = (selected_labels == selected_labels.T).float()

        # final mask of size labels.size(0) x labels.size(0) initialized to zeros
        mask = torch.zeros(labels.size(0), labels.size(
            0), dtype=torch.float32).to(self.device)

        mask[combined_indices.reshape(-1, 1), combined_indices] = mask_partial

        mask += torch.eye(mask.size(0)).to(self.device)
        mask.clamp_(max=1)
        return mask, ratio


class ConSupPrototypeLoss(LightningModule):
    '''
    Supervise contrstive loss function

    temperature: Temperature at which softmax evaluation is done. Temperature
            must be a python scalar or scalar Tensor of numeric dtype.

    contrast_mode: LossContrastMode specifying which views get used as anchors

        'ALL_VIEWS': All the views of all samples are used as anchors 

        'ONE_VIEW': Just the first view of each sample is used as an anchor 
         This view is called the `core` view against
            which other views are contrasted.

    base_temperature: scalar temp

    '''

    def __init__(self,
                 temperature: float = 0.5,
                 contrast_mode: str = "ALL_VIEWS",
                 base_temperature: float = 0.07,

                 negatives_weight=1.0,
                 eps=0.1,


                 eps_0=None,
                 eps_1=None,
                 minority_cls=None,
                 max_epoch=None

                 ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.eps = eps
        self.eps_0 = eps_0
        self.eps_1 = eps_1

        self.prototypes = None
        # self.minority_class = int(minority_class)

        self.negatives_weight = negatives_weight

        if eps_0 is not None and eps_0 == eps_1:
            self.eps = self.eps_0 = self.eps_1 = eps_0

        self.minority_cls = minority_cls
        self.max_epoch = max_epoch

        print(
            f'Loss function: ConLossBszPrototypeLoss {self.base_temperature } {self.temperature } of majority class from numerator')
        print(f" base_temperature {self.base_temperature}")

    def set_prototypes(self, prototypes):
        self.prototypes = prototypes.to(self.device)
        print("setting prototypes")

    def forward(self,
                features,
                labels=None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''

        features: A tensor with a min of 3 dime, structured as [bsz, n_views, ...], where:
                - bsz (int): Batch size, representing the number of samples in a batch.
                - n_views (int): Number of views or augmentations per sample.

        labels: of shape (bsz, 2,) 

        '''

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        num_prototypes = 0
        if self.prototypes is not None:
            num_prototypes = self.prototypes.size(0)

            if num_prototypes != 2:
                raise ValueError('Num of prototypes must be 2')
        if self.prototypes.device != self.device:
            self.prototypes = self.prototypes.to(self.device)

        mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)

        mask_maj = mask_min = mask_sup = None

        if labels is not None:

            labels_dim = labels[:, 1].contiguous().view(-1, 1)
            if labels_dim.shape != (batch_size, 1) or labels.shape != (batch_size, 2):
                raise ValueError(
                    'Number of labels does not match number of features')

            # Create symmetric mask based on label equality, indicating positive pairs
            mask_sup = torch.eq(
                labels_dim, labels_dim.T).float().to(self.device)
            labels_dim = labels_dim.squeeze(1)

            # Create boolean indices for each class
            indices_0 = (labels_dim == 0).to(self.device)
            indices_1 = (labels_dim == 1).to(self.device)

            # Initialize class-specific masks
            mask_0, mask_1 = mask_sup.clone(), mask_sup.clone()

            # Set opposite class indices to 0 in each mask
            mask_0[indices_1] = 0
            mask_0[:, indices_1] = 0
            mask_1[indices_0] = 0
            mask_1[:, indices_0] = 0

            # Validate the 'minority_cls' attribute
            if self.minority_cls is None:
                raise ValueError('minority_cls must be set')

            # Assign masks based on minority class identifier
            mask_min, mask_maj = (
                mask_0, mask_1) if self.minority_cls == 0 else (mask_1, mask_0)

        contrast_count = features.shape[1]  # num of augmentations

        # stack all the augmentations on top of each other
        # first features 0 to bsz and then their augmentations
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.prototypes is not None:
            contrast_feature = torch.cat(
                [contrast_feature, self.prototypes], dim=0)

        if self.contrast_mode == 'ONE_VIEW':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'ALL_VIEWS':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Generate `logits`, the tensor of (temperature-scaled) dot products of the
        # anchor features with all features.

        sims = torch.matmul(anchor_feature, contrast_feature.T)

        anchor_dot_contrast = torch.div(
            sims,
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        # logits_mask is used to get the denominator(positives and negatives)
        logits_mask = torch.scatter(
            # tensor filled with the scalar value 1, with the same size as mask
            torch.ones_like(mask),
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1,
                                                         1).to(self.device),  # index
            0  # fill value
        )

        if mask_maj is not None:
            mask_maj = mask_maj.repeat(anchor_count, contrast_count)
            mask_maj = mask_maj * logits_mask

        if mask_min is not None:
            mask_min = mask_min.repeat(anchor_count, contrast_count)
            mask_min = mask_min * logits_mask

        if mask_sup is not None:
            mask_sup = mask_sup.repeat(anchor_count, contrast_count)
            mask_sup = mask_sup * logits_mask

        mask = mask * logits_mask

        if self.prototypes is not None:

            if labels is None:
                raise ValueError(
                    'labels must be provided if prototypes are provided')

            bsz = labels.shape[0]

            # labels is of shape (bsz, 2,)
            selected_prototypes = labels

            selected_prototypes_mask = selected_prototypes.to(self.device)

            assert selected_prototypes_mask.shape[0] == bsz
            assert selected_prototypes_mask.shape[1] == num_prototypes

            selected_prototypes_mask = torch.cat(
                [selected_prototypes_mask, selected_prototypes_mask], dim=0)  # for the second view

            logits_mask = torch.cat([logits_mask, torch.zeros_like(
                selected_prototypes_mask).to(self.device)], dim=1)

            logits = logits[:-num_prototypes, :]

        # compute log_prob
        # all samples except self contrast case
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - \
            torch.log(self.negatives_weight *
                      (exp_logits.sum(1, keepdim=True)))

        if self.prototypes is not None:
            m_pull = torch.ones_like(
                selected_prototypes_mask, dtype=torch.float32).to(self.device)

            # no pull for samples that are coser to their own prototype than the other prototype
            cond2 = sims[:-num_prototypes, -
                         1] <= (sims[:-num_prototypes, -2] + self.eps_1)

            cond1 = sims[:-num_prototypes, -
                         2] <= (sims[:-num_prototypes, -1] + self.eps_0)

            # Apply conditions and selected_prototypes_mask together
            p2_mask = torch.logical_and(
                cond2, selected_prototypes_mask[:, -1].bool())
            p1_mask = torch.logical_and(
                cond1, selected_prototypes_mask[:, -2].bool())

            selected_prototypes_mask = torch.stack(
                (p1_mask, p2_mask), dim=1).float()

            m_pull = m_pull * selected_prototypes_mask

            mask = torch.cat([mask, m_pull], dim=1)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # print(mean_log_prob_pos)

        if self.base_temperature > 0.0:
            loss = - (self.temperature/self.base_temperature) * \
                mean_log_prob_pos
        else:
            loss = - 1.0 * mean_log_prob_pos

        loss = loss.mean()

        # positives alignment
        labels = torch.arange(batch_size, device=self.device, dtype=torch.long)
        labels = labels+batch_size-1  # Remove sim to self
        labels = torch.cat([labels,  torch.arange(
            batch_size, device=self.device, dtype=torch.long)], dim=0)

        if self.prototypes is not None:
            exp_logits = exp_logits[:, :-num_prototypes]

        # Logits with self sim removed
        clean_logits = exp_logits[~torch.eye(
            batch_size*anchor_count).bool()].view(batch_size*anchor_count, -1)
        ###

        # print(loss)
        return loss, clean_logits.float().detach(), labels.detach()


class SupConLossKCLTSC(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, use_tcl=False, k=3):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        prototypes = self.generate_random_vector_and_negative(
            128)  # Expected shape: [num_classes, feature_dim]
        self.register_buffer('prototypes', prototypes)
        self.k = k
        self.use_tcl = use_tcl

        print(
            f"SupConLossTCL {self.temperature} {self.contrast_mode} {self.base_temperature} {self.use_tcl} {self.k}")

    def generate_random_vector_and_negative(self, d):
        """
        Generates a random vector on the d-dimensional unit hypersphere and its negative.

        Parameters:
        - d (int): The dimension of the hypersphere.

        Returns:
        - torch.Tensor: A 2 x d array where the first row is the random vector
                        and the second row is its negative.
        """
        random_vector = np.random.randn(d)
        norm = np.linalg.norm(random_vector)
        if norm == 0:
            raise ValueError(
                "Generated a zero vector, which cannot be normalized.")
        unit_vector = random_vector / norm
        negative_vector = -unit_vector
        vectors = np.vstack((unit_vector, negative_vector))
        return torch.from_numpy(vectors).float()

    def forward(self, features, labels=None, mask=None,):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        k = self.k
        tcl = self.use_tcl
        num_prototypes = 2

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        anchor_labels = torch.cat([labels, labels], dim=0)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        mask_augs = torch.eye(batch_size, dtype=torch.float32).to(device)
        # print(anchor_labels.shape)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = torch.cat(
            [contrast_feature, self.prototypes], dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits = logits[:-2, :]

        mask = mask.repeat(anchor_count, contrast_count)
        mask_augs = mask_augs.repeat(anchor_count, contrast_count)

        prototype_mask = torch.zeros(
            (2*batch_size, num_prototypes), device=device)
        prototype_mask[torch.arange(2*batch_size), anchor_labels] = 1

        # first_indices = torch.arange(batch_size) + batch_size - 1  # Shape: (A,)

        # # Create indices for the next A rows
        # second_indices = torch.arange(batch_size)  # Shape: (A,)

        # # Concatenate the indices
        # all_indices = torch.cat([first_indices, second_indices])  # Shape: (2A,)

        # # Create one-hot encoded tensor
        # augmentations_mask = torch.nn.functional.one_hot(all_indices, num_classes=batch_size*2).float().to(device)
        # Create sample indices mapping each augmented feature to its original sample
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask_augs = mask_augs * logits_mask

        mask_noself = mask * logits_mask

        augmentations_mask = mask_augs.float()
        # print("augmask",augmentations_mask)

        mask_noself = mask_noself - augmentations_mask.float()

        mask[mask < 1e-6] = 0

        # print("positives",mask.sum(1).mean().cpu().numpy())

        # Randomly select up to k positives per anchor
        num_pos = mask_noself.sum(dim=1)
        # k_max = min(k, num_pos.max().int().item())

        # For anchors with fewer than k positives, adjust k accordingly
        new_pos_mask = torch.zeros_like(mask)
        for i in range(2*batch_size):
            k_i = min(k, int(num_pos[i].item()))

            pos_indices = torch.nonzero(mask_noself[i]).squeeze()
            if len(pos_indices) > 0:
                selected_indices = pos_indices[torch.randperm(len(pos_indices))[
                    :k_i]]
                new_pos_mask[i, selected_indices] = 1

        new_pos_mask = augmentations_mask.float() + new_pos_mask.float()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(new_pos_mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = new_pos_mask * logits_mask

        if tcl:
            mask = torch.cat([mask, prototype_mask.float()], dim=1)

        mask = mask.clamp(max=1)
        if tcl:
            logits_mask = torch.cat([logits_mask, torch.ones(
                2*batch_size, 2).float().to(device)], dim=1)
        if not tcl:
            logits = logits[:, :-2]
        # logits =logits[:,:-2]

        print(prototype_mask.sum(1).mean().cpu().numpy())

        # logits_mask[:,:-2] = 0

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        print(mask_pos_pairs.mean().cpu().numpy())
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConLossDDP(nn.Module):
    '''
    Supervised contrastive loss function

    temperature: Temperature at which softmax evaluation is done. Temperature
            must be a python scalar or scalar Tensor of numeric dtype.

    contrast_mode:  All the views of all samples are used as anchors 

    base_temperature:

    denominator_mode: LossDenominatorMode specifying which positives to include
        in contrastive denominator. See documentation above for more details.
    positives_cap: Integer maximum number of positives *other* than
        augmentations of anchor. Infinite if < 0. Must be multiple of num_views.
        Including augmentations, a maximum of (positives_cap + num_views - 1)
        positives is possible. This parameter modifies the contrastive numerator
        by selecting which positives are present in the summation, and which
        positives contribute to the denominator if denominator_mode ==
        enums.LossDenominatorMode.ALL.


    '''

    def __init__(self,
                 temperature: float = 0.5,
                 contrast_mode: str = "ALL_VIEWS",
                 base_temperature: float = 0.07,
                 min_class: int = None,
                 ratio_remove_majority_class_from_numerator: float = -1.0,
                 ratio_remove_majority_class_from_denominator=-1.0,
                 weighting_positives=-1.0,
                 reweight_global_min_loss=-1.0,

                 ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.majority_class = 0 if int(min_class) == 1 else 1
        self.minority_class = int(min_class)
        self.ratio_remove_majority_class_from_numerator = ratio_remove_majority_class_from_numerator

        self.weighting_positives = weighting_positives

        self.ratio_remove_majority_class_from_denominator = ratio_remove_majority_class_from_denominator

        self.reweight_global_min_loss = reweight_global_min_loss

    def forward(self,
                features,
                labels=None,
                ) -> torch.Tensor:
        '''

        Code is inspired by tensor flow code of the original authors

        features: A tensor with a minimum of 3 dimensions, structured as [bsz, n_views, ...], where:
                - bsz (int): Batch size, representing the number of samples in a batch.
                - n_views (int): Number of views or augmentations per sample.

        labels: 1-dimensional tensor of binary labels, either 0 or 1, with length batch_size:
                - batch_size (int): Number of samples in the batch.

        '''

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        device = features.device

        # Gather features and labels from all devices if using DDP
        if dist.is_available() and dist.is_initialized():
            # No need to create placeholder lists manually now
            gathered_features = dist_f.all_gather(
                features)  # Use dist_f.all_gather
        
            features = torch.cat(gathered_features, dim=0)
          
            if labels is not None:
                gathered_labels = dist_f.all_gather(
                    labels)  # Use dist_f.all_gather
                labels = torch.cat(gathered_labels, dim=0)
       
  
     # Now labels is the gathered tensor

            # features now has shape [batch_size_total, n_views, ...] (across all devices)
            # labels now has shape [batch_size_total] (across all devices)

        else:
            # No DDP, features and labels remain as is
            pass
        batch_size = features.shape[0]
        print("batch size", batch_size)

        # if not isinstance(labels, torch.Tensor) and labels is not None:
        #     labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

        calc_loss_maj_min_sep = False
        mask_unsup = None
        mask_maj = None
        mask_min = None

        ratio = self.ratio_remove_majority_class_from_numerator

        mask_unsup = torch.eye(batch_size, dtype=torch.float32).to(device)

        if labels is None:
            # SimCLR unsupervised loss
            mask = mask_unsup
        elif self.ratio_remove_majority_class_from_numerator > 0.0:
            labels = labels.contiguous().view(-1, 1) if labels is not None else None

            # Ensure labels dimensions match expected shapes if labels are not None
            if labels is not None and labels.shape[0] != batch_size:
                raise ValueError(
                    'Number of labels does not match number of features')

            # SupCon with fewer positives in majority class and no reweighting
            numerator_mask, ratio = self._create_numerator_mask(labels,
                                                                self.ratio_remove_majority_class_from_numerator,
                                                                self.minority_class, self.majority_class, device=device)
            mask = numerator_mask
        else:
            # Create masks for supervised loss with handling for majority and minority class balance
            min_indices = labels == self.minority_class
            maj_indices = labels == self.majority_class

            labels = labels.contiguous().view(-1, 1) if labels is not None else None

            # Ensure labels dimensions match expected shapes if labels are not None
            if labels is not None and labels.shape[0] != batch_size:
                raise ValueError(
                    'Number of labels does not match number of features')

            mask = torch.eq(labels, labels.T).float().to(device)
            mask_maj, mask_min = mask.clone(), mask.clone()

            mask_maj[min_indices] = 0.0
            mask_maj[:, min_indices] = 0.0
            mask_min[maj_indices] = 0.0
            mask_min[:, maj_indices] = 0.0

        contrast_count = features.shape[1]  # num of augmentations
        # stack all the augmentations on top of each other
        # first features 0 to bsz and then their augmentations
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'ALL_VIEWS':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        # Generate `logits`, the tensor of (temperature-scaled) dot products of the
        # anchor features with all features.

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        if mask_maj is not None:
            mask_maj = mask_maj.repeat(anchor_count, contrast_count)

        if mask_min is not None:
            mask_min = mask_min.repeat(anchor_count, contrast_count)
        if mask_unsup is not None:
            mask_unsup = mask_unsup.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        # logits_mask is used to get the denominator(positives and negatives)
        logits_mask = torch.scatter(
            # tensor filled with the scalar value 1, with the same size as mask
            torch.ones_like(mask),
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1,
                                                         1).to(device),  # index
            0  # fill value
        )

        mask = mask * logits_mask
        if mask_unsup is not None:
            mask_unsup = mask_unsup * logits_mask

        if mask_min is not None:
            mask_maj = mask_maj * logits_mask
            mask_min = mask_min * logits_mask

        # mask out some of the majority class logits from the denominator
        # not used in paper
        if self.ratio_remove_majority_class_from_denominator > 0.0:
            logits_mask = self._remove_majority_class_from_denominator_mask(
                logits_mask, labels, anchor_count, contrast_count)
        # compute log_prob
        # all samples except self contrast case
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / \
            (mask.sum(1))  # num stability

        if self.base_temperature > 0.0:
            loss = - self.temperature/self.base_temperature * mean_log_prob_pos
        else:
            loss = - 1.0 * mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()

        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        labels = labels+batch_size-1  # Remove sim to self
        labels = torch.cat([labels,  torch.arange(
            batch_size, device=device, dtype=torch.long)], dim=0)

        # Logits with self sim removed
        clean_logits = exp_logits[~torch.eye(
            batch_size*anchor_count).bool()].view(batch_size*anchor_count, -1)
        ####

        return loss, clean_logits, labels

    def _remove_majority_class_from_denominator_mask(self, logits_mask, labels, anchor_count, contrast_count):
        ratio = self.ratio_remove_majority_class_from_denominator  # ratio of maj to remove
        majority_indices = torch.where(labels == self.majority_class)[0]

        # select ratio number of indices from the majority class at random
        selected_majority_indices = torch.randperm(majority_indices.size(0))[
            :int(ratio * majority_indices.size(0))]
        selected_majority_indices = majority_indices[selected_majority_indices]
        mask = torch.ones((labels.size(0), labels.size(0)),
                          device=labels.device)

        mask[:, selected_majority_indices] = 0
        mask = mask.repeat(anchor_count, contrast_count)

        # Move the tensor back to the original device.
        return logits_mask * mask

    def _create_numerator_mask(self, labels, ratio, minority_class, majority_class, warm_up=15, device=None):

        ratio = 1.0 - ratio
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"Ratio must be between 0 and 1, but is {ratio}")

        minority_indices = torch.where(labels == minority_class)[0]
        majority_indices = torch.where(labels == majority_class)[0]

        selected_majority_indices = torch.randperm(majority_indices.size(0))[
            :int(ratio * majority_indices.size(0))]

        selected_majority_indices = majority_indices[selected_majority_indices]
        combined_indices = torch.cat(
            [minority_indices, selected_majority_indices])

        selected_labels = labels[combined_indices].view(-1, 1)
        mask_partial = (selected_labels == selected_labels.T).float()

        # final mask of size labels.size(0) x labels.size(0) initialized to zeros
        mask = torch.zeros(labels.size(0), labels.size(
            0), dtype=torch.float32).to(device)

        mask[combined_indices.reshape(-1, 1), combined_indices] = mask_partial

        mask += torch.eye(mask.size(0)).to(device)
        mask.clamp_(max=1)
        return mask, ratio


class ConSupPrototypeLossDDP(nn.Module):
    '''
    Supervised contrastive loss function

    temperature: Temperature at which softmax evaluation is done. Temperature
            must be a python scalar or scalar Tensor of numeric dtype.

    contrast_mode: LossContrastMode specifying which views get used as anchors

        'ALL_VIEWS': All the views of all samples are used as anchors 

        'ONE_VIEW': Just the first view of each sample is used as an anchor 
         This view is called the `core` view against
            which other views are contrasted.

    base_temperature: scalar temp

    '''

    def __init__(self,
                 temperature: float = 0.5,
                 contrast_mode: str = "ALL_VIEWS",
                 base_temperature: float = 0.07,

                 negatives_weight=1.0,
                 eps=0.1,


                 eps_0=None,
                 eps_1=None,
                 minority_cls=None,
                 max_epoch=None

                 ) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.eps = eps
        self.eps_0 = eps_0
        self.eps_1 = eps_1

        self.prototypes = None
        # self.minority_class = int(minority_class)

        self.negatives_weight = negatives_weight

        if eps_0 is not None and eps_0 == eps_1:
            self.eps = self.eps_0 = self.eps_1 = eps_0

        self.minority_cls = minority_cls
        self.max_epoch = max_epoch

        print(
            f'Loss function: ConLossBszPrototypeLoss {self.base_temperature } {self.temperature } of majority class from numerator')
        print(f" base_temperature {self.base_temperature}")

    def set_prototypes(self, prototypes):
        self.prototypes = prototypes#.to(device)
        print("setting prototypes")

    def forward(self,
                features,
                labels=None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''

        features: A tensor with a minimum of 3 dimensions, structured as [bsz, n_views, ...], where:
                - bsz (int): Batch size, representing the number of samples in a batch.
                - n_views (int): Number of views or augmentations per sample.

        labels: of shape (bsz, 2,) 

        '''

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            
        device = features.device

        # Gather features and labels from all devices if using DDP
        if dist.is_available() and dist.is_initialized():
            # No need to create placeholder lists manually now
            gathered_features = dist_f.all_gather(
                features)  # Use dist_f.all_gather
        
            features = torch.cat(gathered_features, dim=0)
          
            if labels is not None:
                gathered_labels = dist_f.all_gather(
                    labels)  # Use dist_f.all_gather
                labels = torch.cat(gathered_labels, dim=0)
       
  
     # Now labels is the gathered tensor

            # features now has shape [batch_size_total, n_views, ...] (across all devices)
            # labels now has shape [batch_size_total] (across all devices)

        else:
            # No DDP, features and labels remain as is
            pass
        batch_size = features.shape[0]
        print("batch size", batch_size)

        num_prototypes = 0
        if self.prototypes is not None:
            num_prototypes = self.prototypes.size(0)

            if num_prototypes != 2:
                raise ValueError('Number of prototypes must be 2')
            if self.prototypes.device != device:
                self.prototypes = self.prototypes.to(device)

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        mask_maj = mask_min = mask_sup = None

        if labels is not None:

            labels_dim = labels[:, 1].contiguous().view(-1, 1)
            if labels_dim.shape != (batch_size, 1) or labels.shape != (batch_size, 2):
                raise ValueError(
                    'Number of labels does not match number of features')

            # Create symmetric mask based on label equality, indicating positive pairs
            mask_sup = torch.eq(
                labels_dim, labels_dim.T).float().to(device)
            labels_dim = labels_dim.squeeze(1)

            # Create boolean indices for each class
            indices_0 = (labels_dim == 0).to(device)
            indices_1 = (labels_dim == 1).to(device)

            # Initialize class-specific masks
            mask_0, mask_1 = mask_sup.clone(), mask_sup.clone()

            # Set opposite class indices to 0 in each mask
            mask_0[indices_1] = 0
            mask_0[:, indices_1] = 0
            mask_1[indices_0] = 0
            mask_1[:, indices_0] = 0

            # Validate the 'minority_cls' attribute
            if self.minority_cls is None:
                raise ValueError('minority_cls must be set')

            # Assign masks based on minority class identifier
            mask_min, mask_maj = (
                mask_0, mask_1) if self.minority_cls == 0 else (mask_1, mask_0)

        contrast_count = features.shape[1]  # num of augmentations

        # stack all the augmentations on top of each other
        # first features 0 to bsz and then their augmentations
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.prototypes is not None:
            contrast_feature = torch.cat(
                [contrast_feature, self.prototypes], dim=0)

        if self.contrast_mode == 'ONE_VIEW':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'ALL_VIEWS':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Generate `logits`, the tensor of (temperature-scaled) dot products of the
        # anchor features with all features.

        sims = torch.matmul(anchor_feature, contrast_feature.T)

        anchor_dot_contrast = torch.div(
            sims,
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        # logits_mask is used to get the denominator(positives and negatives)
        logits_mask = torch.scatter(
            # tensor filled with the scalar value 1, with the same size as mask
            torch.ones_like(mask),
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1,
                                                         1).to(device),  # index
            0  # fill value
        )

        if mask_maj is not None:
            mask_maj = mask_maj.repeat(anchor_count, contrast_count)
            mask_maj = mask_maj * logits_mask

        if mask_min is not None:
            mask_min = mask_min.repeat(anchor_count, contrast_count)
            mask_min = mask_min * logits_mask

        if mask_sup is not None:
            mask_sup = mask_sup.repeat(anchor_count, contrast_count)
            mask_sup = mask_sup * logits_mask

        mask = mask * logits_mask

        if self.prototypes is not None:

            if labels is None:
                raise ValueError(
                    'labels must be provided if prototypes are provided')

            bsz = labels.shape[0]

            # labels is of shape (bsz, 2,)
            selected_prototypes = labels

            selected_prototypes_mask = selected_prototypes.to(device)

            assert selected_prototypes_mask.shape[0] == bsz
            assert selected_prototypes_mask.shape[1] == num_prototypes

            selected_prototypes_mask = torch.cat(
                [selected_prototypes_mask, selected_prototypes_mask], dim=0)  # for the second view

            logits_mask = torch.cat([logits_mask, torch.zeros_like(
                selected_prototypes_mask).to(device)], dim=1)

            logits = logits[:-num_prototypes, :]

        # compute log_prob
        # all samples except self contrast case
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - \
            torch.log(self.negatives_weight *
                      (exp_logits.sum(1, keepdim=True)))

        if self.prototypes is not None:
            m_pull = torch.ones_like(
                selected_prototypes_mask, dtype=torch.float32).to(device)

            # no pull for samples that are closer to their own prototype than the other prototype
            cond2 = sims[:-num_prototypes, -
                         1] <= (sims[:-num_prototypes, -2] + self.eps_1)

            cond1 = sims[:-num_prototypes, -
                         2] <= (sims[:-num_prototypes, -1] + self.eps_0)

            # Apply conditions and selected_prototypes_mask together
            p2_mask = torch.logical_and(
                cond2, selected_prototypes_mask[:, -1].bool())
            p1_mask = torch.logical_and(
                cond1, selected_prototypes_mask[:, -2].bool())

            selected_prototypes_mask = torch.stack(
                (p1_mask, p2_mask), dim=1).float()

            m_pull = m_pull * selected_prototypes_mask

            mask = torch.cat([mask, m_pull], dim=1)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        if self.base_temperature > 0.0:
            loss = - (self.temperature/self.base_temperature) * \
                mean_log_prob_pos
        else:
            loss = - 1.0 * mean_log_prob_pos

        loss = loss.mean()

        # positives alignment
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        labels = labels+batch_size-1  # Remove sim to self
        labels = torch.cat([labels,  torch.arange(
            batch_size, device=device, dtype=torch.long)], dim=0)

        if self.prototypes is not None:
            exp_logits = exp_logits[:, :-num_prototypes]

        # Logits with self sim removed
        clean_logits = exp_logits[~torch.eye(
            batch_size*anchor_count).bool()].view(batch_size*anchor_count, -1)
        ###

        return loss, clean_logits.float().detach(), labels.detach()
