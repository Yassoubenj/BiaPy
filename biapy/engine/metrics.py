import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from torchmetrics import JaccardIndex
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms.functional import resize
import torchvision.transforms as T
from pytorch_msssim import SSIM
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from typing import Dict, Optional, List

# def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#     """
#     Mean Squared Error loss, voxel-wise, reduction='mean'.
#     predictions: (B, 1, Z, Y, X)
#     targets:     (B, 1, Z, Y, X)
#     """
#     return F.mse_loss(predictions, targets)

def jaccard_index_numpy(y_true, y_pred):
    """
    Define Jaccard index.

    Parameters
    ----------
    y_true : N dim Numpy array
        Ground truth masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
        ``(volume_number, z, x, y, channels)`` for 3D volumes.

    y_pred : N dim Numpy array
        Predicted masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
        ``(volume_number, z, x, y, channels)`` for 3D volumes.

    Returns
    -------
    jac : float
        Jaccard index value.
    """

    if y_true.ndim != y_pred.ndim:
        raise ValueError("Dimension mismatch: {} and {} provided".format(y_true.shape, y_pred.shape))

    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac


def jaccard_index_numpy_without_background(y_true, y_pred):
    """
    Define Jaccard index excluding the background class (first channel).

    Parameters
    ----------
    y_true : N dim Numpy array
        Ground truth masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
        ``(volume_number, z, x, y, channels)`` for 3D volumes.

    y_pred : N dim Numpy array
        Predicted masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
        ``(volume_number, z, x, y, channels)`` for 3D volumes.

    Returns
    -------
    jac : float
        Jaccard index value.
    """

    if y_true.ndim != y_pred.ndim:
        raise ValueError("Dimension mismatch: {} and {} provided".format(y_true.shape, y_pred.shape))

    TP = np.count_nonzero(y_pred[..., 1:] * y_true[..., 1:])
    FP = np.count_nonzero(y_pred[..., 1:] * (y_true[..., 1:] - 1))
    FN = np.count_nonzero((y_pred[..., 1:] - 1) * y_true[..., 1:])

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac


def weight_binary_ratio(target):
    if torch.max(target) == torch.min(target):
        return torch.ones_like(target, dtype=torch.float32)

    # Generate weight map by balancing the foreground and background.
    min_ratio = 5e-2
    label = target.clone()  # copy of target label

    label = (label != 0).double()  # foreground

    ww = label.sum() / torch.prod(torch.tensor(label.shape, dtype=torch.double))

    ww = torch.clamp(ww, min=min_ratio, max=1 - min_ratio)

    weight_factor = max(ww, 1 - ww) / min(ww, 1 - ww)  # type: ignore

    # Case 1 -- Affinity Map
    # In that case, ww is large (i.e., ww > 1 - ww), which means the high weight
    # factor should be applied to background pixels.

    # Case 2 -- Contour Map
    # In that case, ww is small (i.e., ww < 1 - ww), which means the high weight
    # factor should be applied to foreground pixels.

    if ww > 1 - ww:
        # Switch when foreground is the dominant class.
        label = 1 - label
    weight = weight_factor * label + (1 - label)

    return weight.float()


class jaccard_index:
    def __init__(self, num_classes, device, t=0.5, model_source="biapy"):
        """
        Define Jaccard index.

        Parameters
        ----------
        num_classes : int
            Number of classes.

        device : Torch device
            Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps",
            "xpu", "xla" or "meta".

        t : float, optional
            Threshold to be applied.

        model_source : str, optional
            Source of the model. It can be "biapy", "bmz" or "torchvision".
        """
        self.model_source = model_source
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = device
        self.num_classes = num_classes
        self.t = t

        if self.num_classes > 2:
            self.jaccard = JaccardIndex(task="multiclass", threshold=self.t, num_classes=self.num_classes).to(
                self.device, non_blocking=True
            )
        else:
            self.jaccard = JaccardIndex(task="binary", threshold=self.t, num_classes=self.num_classes).to(
                self.device, non_blocking=True
            )

    def __call__(self, y_pred, y_true):
        """
        Calculate CrossEntropyLoss.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor
            Predicted masks.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        # If image shape has changed due to TorchVision or BMZ preprocessing then the mask needs
        # to be resized too
        if self.model_source == "torchvision":
            if y_pred.shape[-2:] != y_true.shape[-2:]:
                y_true = resize(
                    y_true,
                    size=y_pred.shape[-2:],
                    interpolation=T.InterpolationMode("nearest"),
                )
            if torch.max(y_true) > 1 and self.num_classes <= 2:
                y_true = (y_true / 255).type(torch.long)
        # For those cases that are predicting 2 channels (binary case) we adapt the GT to match.
        # It's supposed to have 0 value as background and 1 as foreground
        elif self.model_source == "bmz" and self.num_classes <= 2 and y_pred.shape[1] != y_true.shape[1]:
            y_true = torch.cat((1 - y_true, y_true), 1)

        if self.num_classes > 2:
            if y_pred.shape[1] > 1:
                y_true = y_true.squeeze()
            if len(y_pred.shape) - 2 == len(y_true.shape):
                y_true = y_true.unsqueeze(0)

        return self.jaccard(y_pred, y_true)


class multiple_metrics:
    def __init__(self, num_classes, metric_names, device, val_to_ignore: Optional[int] = None, model_source="biapy"):
        """
        Define instance segmentation workflow metrics.

        Parameters
        ----------
        num_classes : int
            Number of classes.

        metric_names : list of str
            Names of the metrics to use.

        device : Torch device
            Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps",
            "xpu", "xla" or "meta".

        model_source : str, optional
            Source of the model. It can be "biapy", "bmz" or "torchvision".
        """

        self.num_classes = num_classes
        self.metric_names = metric_names
        self.device = device
        self.model_source = model_source
        self.val_to_ignore = val_to_ignore

        self.metric_func = []
        for i in range(len(metric_names)):
            if "IoU (classes)" in metric_names[i]:
                loss_func = JaccardIndex(task="multiclass", threshold=0.5, num_classes=self.num_classes, ignore_index=self.val_to_ignore).to(
                    self.device, non_blocking=True
                )
            elif "IoU" in metric_names[i]:
                loss_func = JaccardIndex(task="binary", threshold=0.5, num_classes=2, ignore_index=self.val_to_ignore).to(self.device, non_blocking=True)
            elif metric_names[i] == "L1 (distance channel)":
                loss_func = torch.nn.L1Loss()

            self.metric_func.append(loss_func)

    def __call__(self, y_pred, y_true):
        """
        Calculate metrics.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor or list of Tensors
            Prediction.

        Returns
        -------
        dict : dict
            Metrics and their values.
        """
        # Check multi-head
        if isinstance(y_pred, list):
            num_channels = y_pred[0].shape[1] + 1
            _y_pred = y_pred[0]
            _y_pred_class = torch.argmax(y_pred[1], dim=1)
        else:
            num_channels = y_pred.shape[1]
            _y_pred = y_pred
            _y_pred_class = y_pred[:, -1]

        # If image shape has changed due to TorchVision or BMZ preprocessing then the mask needs
        # to be resized too
        if self.model_source == "torchvision":
            if _y_pred.shape[-2:] != y_true.shape[-2:]:
                y_true = resize(
                    y_true,
                    size=_y_pred.shape[-2:],
                    interpolation=T.InterpolationMode("nearest"),
                )
            if torch.max(y_true) > 1 and self.num_classes <= 2:
                y_true = (y_true / 255).type(torch.long)

        res_metrics = {}
        for i in range(num_channels):
            if self.metric_names[i] not in res_metrics:
                res_metrics[self.metric_names[i]] = []
            # Measure metric
            if self.metric_names[i] == "IoU (classes)":
                res_metrics[self.metric_names[i]].append(self.metric_func[i](_y_pred_class, y_true[:, 1]))
            else:
                res_metrics[self.metric_names[i]].append(self.metric_func[i](_y_pred[:, i], y_true[:, 0]))

        # Mean of same metric values
        for key, value in res_metrics.items():
            if len(value) > 1:
                res_metrics[key] = torch.mean(torch.as_tensor(value))
            else:
                res_metrics[key] = torch.as_tensor(value[0])
        return res_metrics


class CrossEntropyLoss_wrapper:
    def __init__(self, num_classes, multihead=False, model_source="biapy", class_rebalance=False):
        """
        Wrapper to Pytorch's CrossEntropyLoss.

        Parameters
        ----------
        num_classes : int
            Number of classes.

        multihead : bool, optional
            For multihead predictions e.g. points + classification in detection.

        model_source : str, optional
            Source of the model. It can be "biapy", "bmz" or "torchvision".

        class_rebalance: bool, optional
            Whether to reweight classes (inside loss function) or not.
        """
        self.model_source = model_source
        self.multihead = multihead
        self.num_classes = num_classes
        self.class_rebalance = class_rebalance
        if num_classes <= 2:
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.class_channel_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        """
        Calculate CrossEntropyLoss.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor
            Predicted masks.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        if self.multihead:
            _y_pred = y_pred[0]
            _y_pred_class = y_pred[1]
            assert (
                y_true.shape[1] == 2
            ), f"In multihead setting the ground truth is expected to have 2 channels. Provided {y_true.shape}"
        else:
            _y_pred = y_pred

        # If image shape has changed due to TorchVision or BMZ preprocessing then the mask needs
        # to be resized too
        if self.model_source == "torchvision":
            if _y_pred.shape[-2:] != y_true.shape[-2:]:
                y_true = resize(
                    y_true,
                    size=_y_pred.shape[-2:],
                    interpolation=T.InterpolationMode("nearest"),
                )
            if torch.max(y_true) > 1 and self.num_classes <= 2:
                y_true = (y_true / 255).type(torch.float32)
        # For those cases that are predicting 2 channels (binary case) we adapt the GT to match.
        # It's supposed to have 0 value as background and 1 as foreground
        elif self.model_source == "bmz" and self.num_classes <= 2 and _y_pred.shape[1] != y_true.shape[1]:
            y_true = torch.cat((1 - y_true, y_true), 1)

        if self.class_rebalance:
            if self.multihead:
                weight_mask = weight_binary_ratio(y_true[:, 0])
                loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight_mask)
            else:
                weight_mask = weight_binary_ratio(y_true)
                if self.num_classes <= 2:
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight_mask)
                else:
                    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_mask)
        else:
            loss_fn = self.loss

        if self.multihead:
            loss = loss_fn(_y_pred[:, 0], y_true[:, 0]) + self.class_channel_loss(
                _y_pred_class, y_true[:, -1].type(torch.long)
            )
        else:
            if self.num_classes <= 2:
                loss = loss_fn(_y_pred, y_true)
            else:
                loss = loss_fn(_y_pred, y_true[:, 0].type(torch.long))

        return loss


class DiceLoss(nn.Module): #mais à quoi correspond ces target dans biapy ? a quoi correspond les input : des logit donc on fait sigmoid et pour target on fait rien ? 
    """
    Based on `Kaggle <https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch>`_.
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum() #dans monai il fait un slicing ici 
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
    
def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """
    Perform soft erosion on the input image
    Adapted from:
    https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L6
    """
    if img.dim() == 4:
        p1 = -(F.max_pool2d(-img, (3, 1), (1, 1), (1, 0)))
        p2 = -(F.max_pool2d(-img, (1, 3), (1, 1), (0, 1)))
        return torch.min(p1, p2)
    elif img.dim() == 5:
        p1 = -(F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0)))
        p2 = -(F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0)))
        p3 = -(F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1)))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """
    Perform soft dilation on the input image
    Adapted from:
    https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L18
    """
    if img.dim() == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif img.dim() == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to perform soft opening on the input image
    """
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int) -> torch.Tensor:
    """
    Perform soft skeletonization on the input image
    Adapted from:
    https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L29
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel
    
class SoftclDiceLoss(nn.Module):
    """
    Soft clDice loss adapted au style Biapy.

    Args:
        iter_ (int): nombre d'itérations pour la squelettisation.
        smooth (float): paramètre de lissage.
    """
    def __init__(self, iter_: int = 3, smooth: float = 1.0):
        super(SoftclDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self._printed=False

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Appliquer la sigmoïde comme dans DiceLoss
        inputs = F.sigmoid(inputs) 
    
        # Calcul des squelettes “soft”
        skel_pred = soft_skel(inputs, self.iter)
        skel_true = soft_skel(targets, self.iter)

        # Aplatir tous les tenseurs
        skel_pred_flat = skel_pred.view(-1)
        skel_true_flat = skel_true.view(-1)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # Topological precision
        tprec_num = (skel_pred_flat * targets_flat).sum()
        tprec = (tprec_num + self.smooth) / (skel_pred_flat.sum() + self.smooth)

        # Topological sensitivity
        tsens_num = (skel_true_flat * inputs_flat).sum()
        tsens = (tsens_num + self.smooth) / (skel_true_flat.sum() + self.smooth)

        # clDice
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice
        
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# #Cldice from scratch 
# def _soft_erode(x: torch.Tensor) -> torch.Tensor:
#     """Soft morphological erosion (2D ou 3D)."""
#     ndim = x.ndim - 2 # 2 => 2D, 3 => 3D
#     pool = F.max_pool2d if ndim == 2 else F.max_pool3d
#     # deux erosions anisotropes puis min :
#     if ndim == 2: # (H,W)
#         k1, k2 = (3, 1), (1, 3)
#     else: # (D,H,W)
#         k1, k2 = (3, 1, 1), (1, 3, 3)
#     p1 = -pool(-x, k1, stride=1, padding=[k//2 for k in k1])
#     p2 = -pool(-x, k2, stride=1, padding=[k//2 for k in k2])
#     return torch.minimum(p1, p2)

# def _soft_skel(x: torch.Tensor, it: int = 3) -> torch.Tensor:
#     """Approximation différentiable du squelette."""
#     skel = torch.zeros_like(x)
#     for _ in range(it):
#         eroded = _soft_erode(x)
#         opened = F.max_pool3d(eroded, 3, 1, 1) if x.ndim == 5 else \
#                  F.max_pool2d(eroded, 3, 1, 1)
#         skel = skel + F.relu(opened - eroded)
#         x = eroded
#     return skel

# class SoftClDiceLoss(torch.nn.Module):
#     """
#     • Accepte : y_pred logits ou probas de shape (B, C, …).
#     • Tolère C = 1 (binaire) ou >1 (multiclasses).
#     • from_logits : appliquer sigmoïde/softmax interne si True.
#     """
#     def __init__(self, iter_: int = 3, smooth: float = 1.,
#                 from_logits: bool = True):
#         super().__init__()
#         self.iter_ = iter_
#         self.smooth = smooth
#         self.from_logits = from_logits

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         """
#         y_true : (B,1,…) ou (B,C,…).  
#         y_pred : (B,1,…) ou (B,C,…).
#         """
#         # Activation de la prediction ---------------------------
#         if self.from_logits:
#             print(y_pred.shape[1]) #la normalement ça rend 1
#             if y_pred.shape[1] == 1:           # binaire c'est notre cas ( même si n_classe = 2)
#                 y_pred = torch.sigmoid(y_pred)
#             else:                              # multi-classe
#                 y_pred = F.softmax(y_pred, dim=1)

#         # Harmonise GT ---------------------------------------------------------------
#         if y_true.dtype != torch.float32:
#             y_true = y_true.float()

#         if y_true.shape[1] == 1 and y_pred.shape[1] > 1:
#             # one-hot GT si sortie multi-classe
#             y_true = F.one_hot(y_true.squeeze(1).long(), num_classes=y_pred.shape[1]
#                             ).permute(0,4,1,2,3).float()

#         # Si y_pred n’a qu’un canal, crée un « canal background » (1-p) qu'on va ignorer après on se concentre sur foreground 
#         if y_pred.shape[1] == 1:
#             y_pred = torch.cat([1 - y_pred, y_pred], dim=1)
#             y_true = torch.cat([1 - y_true, y_true], dim=1)

#         C = y_pred.shape[1] # nb de classes (≥2 désormais)
#         losses = []

#         for c in range(1, C):        # on ignore explicitement le background (c=0)
#             p, t = y_pred[:, c], y_true[:, c]

#             sk_p, sk_t = _soft_skel(p, self.iter_), _soft_skel(t, self.iter_)

#             tprec = (torch.sum(sk_p * t) + self.smooth) / \
#                     (torch.sum(sk_p) + self.smooth)
#             tsens = (torch.sum(sk_t * p) + self.smooth) / \
#                     (torch.sum(sk_t) + self.smooth)

#             cl = (2.0 * tprec * tsens) / (tprec + tsens + 1e-7)
#             losses.append(1.0 - cl)

#         return torch.mean(torch.stack(losses))


#CLDICE FROM MONAI 




# class SoftclDiceLoss(_Loss):
#     def __init__(
#         self,
#         iter_: int = 3,
#         smooth: float = 1.0,
#         num_classes: int | None = None,
#     ) -> None:
#         super().__init__()
#         self.iter = iter_
#         self.smooth = smooth
#         self.num_classes = num_classes  
        
#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         #y_pred : logit 
#         #y_true : indice 

#         #1) logits → probabilités
#         C_pred = y_pred.size(1) #les prediction : c'est le num classe ( à modifier dans semantic_seg N_classes pour qu'il soit à 2 et non plus à 1)
#         print(f"[DEBUG] C_pred = {C_pred}")
#         y_true = y_true.squeeze(1)
#         print(y_pred.dim())

#         prob = F.softmax(y_pred, dim=1)
#         print(prob.dim())
#         # if C_pred == 1 :
#         #     #segmentation binaire : 
#         #     p_fg = torch.sigmoid(y_pred) #foreground
#         #     p_bg = 1.0 - p_fg #background
#         #     prob = torch.cat([p_bg,p_fg], dim=1)
#         #     #C=2
#         # else :
#         #     prob = F.softmax(y_pred, dim=1)
#         #     C=C_pred

#         # 2) indices → one-hot
#         print("y_pred type:", type(y_pred), y_pred.shape)
#         print("y_true type:", type(y_true), y_true.shape)
#         if y_true.dim() == prob.dim():
#             y_true = y_true.squeeze(1)

#         C_pred = y_true.size(1)
#         print(f"[DEBUG] C_pred = {C_pred}")
#         y_true = y_true.long()
#         y_true_oh = F.one_hot(y_true, num_classes=1).float().movedim(-1,1) 

#         # if y_true.dim() == 4 and y_true.size(1) == 1:
#         #     y_true = y_true.squeeze(1)

        
#         # A detag one-hot + permutation 
#         # if C_pred == 1 :
#         # # #binaire : on empile mannuellement background et foreground
#         #      mask_bg = (y_true==0).long()
#         #      mask_fg = (y_true==1).long()
#         #      y_true_oh = torch.stack([mask_bg,mask_fg], dim=1).float()
#         # else : 
#         #      y_true_oh = F.one_hot(y_true, num_classes=self.num_classes).float().movedim(-1,1)
        
#         #y_true_oh = y_true_oh.permute(0, 3, 1, 2).float() [B,C,H,W]
#         #on devrait avoir deux canaux : foreground et bakground ? 
#         #dans monai on ignore background et on compare que foreground donc canal 1 

#         # 3) on appelle le pipeline clDice original sur des tenseurs valides
#         skel_pred = soft_skel(prob, self.iter)
#         skel_true = soft_skel(y_true_oh, self.iter)
#         #mettre le canal 1 après
#         tprec = (
#             torch.sum(torch.multiply(skel_pred, y_true_oh)[:, 0:, ...]) + self.smooth
#         ) / (torch.sum(skel_pred[:, 0:, ...]) + self.smooth)

#         tsens = (
#             torch.sum(torch.multiply(skel_true, prob)[:, 0:, ...]) + self.smooth
#         ) / (torch.sum(skel_true[:, 0:, ...]) + self.smooth)

#         return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)



# class SoftclDiceLoss(_Loss):
#     """
#     Compute the Soft clDice loss defined in:
#     Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
#     for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)
#     """
#     def __init__(self, iter_: int = 3, smooth: float = 1.0) -> None:
#         super().__init__()
#         self.iter = iter_
#         self.smooth = smooth

#     def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
#         skel_pred = soft_skel(y_pred, self.iter) #devient nul 
#         skel_true = soft_skel(y_true, self.iter) #aussi
#         tprec = (
#             torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth
#         ) / (
#             torch.sum(skel_pred[:, 1:, ...]) + self.smooth
#         )
#         tsens = (
#             torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth
#         ) / (
#             torch.sum(skel_true[:, 1:, ...]) + self.smooth
#         )
#         return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

# class SoftclDiceLossWrapper(_Loss):
#     """
#     Wrapper pour SoftclDiceLoss qui :
#       - prend en entrée des logits [B, C, H, W] et des masks indice [B, H, W] ou [B,1,H,W]
#       - fait softmax sur les logits
#       - convertit les masks en one-hot
#       - appelle SoftclDiceLoss(y_true_oh, prob)
#     """
#     def __init__(self, iter_: int, smooth: float, num_classes: int) -> None:
#         super().__init__()
#         self.cldice = SoftclDiceLoss(iter_, smooth)
#         self.num_classes = num_classes

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         prob = F.softmax(y_pred, dim=1) #obtenir les proba

#         # 2) indices → one-hot [B, C, H, W]
#         if y_true.dim() == 4 and y_true.size(1) == 1:
#             y_true = y_true.squeeze(1)
#         y_true = y_true.long()  
#         y_true_oh = F.one_hot(y_true, num_classes=self.num_classes)  # [B, H, W, C]
#         y_true_oh = y_true_oh.permute(0, 3, 1, 2).float()             # [B, C, H, W]

#         # 3) appel du SoftclDiceLoss “pur”
#         return self.cldice(y_true_oh, prob)


# class SoftDiceclDiceLoss(_Loss):
#     """
#     Combine Soft Dice and Soft clDice losses
#     """
#     def __init__(
#         self, iter_: int = 3, alpha: float = 0.5, smooth: float = 1.0
#     ) -> None:
#         super().__init__()
#         self.iter = iter_
#         self.alpha = alpha
#         self.smooth = smooth

#     def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
#         dice_loss = soft_dice(y_true, y_pred, self.smooth)
#         skel_pred = soft_skel(y_pred, self.iter)
#         skel_true = soft_skel(y_true, self.iter)
#         tprec = (
#             torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth
#         ) / (
#             torch.sum(skel_pred[:, 1:, ...]) + self.smooth
#         )
#         tsens = (
#             torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth
#         ) / (
#             torch.sum(skel_true[:, 1:, ...]) + self.smooth
#         )
#         cldice_loss = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
#         return (1.0 - self.alpha) * dice_loss + self.alpha * cldice_loss

class DiceBCELoss(nn.Module):
    """
    Based on `Kaggle <https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch>`_.
    """

    def __init__(self, w_dice=0.5, w_bce=0.5):
        super(DiceBCELoss, self).__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = (BCE * self.w_bce) + (dice_loss * self.w_dice)

        return Dice_BCE
class SoftclDiceBCELoss(nn.Module):
    """
    Combines Binary Cross-Entropy and Soft clDice loss for
    binary segmentation (1 canal).

    Args:
        w_cldice (float): poids du terme clDice.
        w_bce    (float): poids du terme BCE.
        iter_    (int):   nombre d’itérations pour soft_skel().
        smooth   (float): paramètre de lissage.
    """
    def __init__(self, w_cldice: float = 0.0, w_bce: float = 1.0,
                iter_: int = 3, smooth: float = 1.0):
        super(SoftclDiceBCELoss, self).__init__()
        self.w_cldice = w_cldice
        self.w_bce    = w_bce
        self.iter     = iter_
        self.smooth   = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: logits non bornés → proba [0,1]
        inputs = F.sigmoid(inputs)

        # aplatir pour le BCE
        inputs_flat  = inputs.view(-1)
        targets_flat = targets.view(-1)

        # 1) BCE
        bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction="mean")

        # 2) Soft clDice
        skel_pred     = soft_skel(inputs, self.iter)
        skel_true     = soft_skel(targets, self.iter)
        pred_flat_s   = skel_pred.view(-1)
        true_flat_s   = skel_true.view(-1)

        tprec_num = (pred_flat_s * targets_flat).sum()
        tprec     = (tprec_num + self.smooth) / (pred_flat_s.sum() + self.smooth)

        tsens_num = (true_flat_s * inputs_flat).sum()
        tsens     = (tsens_num + self.smooth) / (true_flat_s.sum() + self.smooth)

        cldice_loss = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

        # 3) combinaison
        loss = self.w_bce * bce + self.w_cldice * cldice_loss
        return loss

class SoftclDiceFocalLoss(nn.Module):
    """
    Combine Binary Focal Loss (single-channel) and Soft clDice
    for binary segmentation (1-canal).

    Args:
        w_focal   (float): poids du terme Focal Loss.
        w_cldice  (float): poids du terme Soft clDice.
        iter_     (int):   nombre d’itérations pour soft_skel().
        smooth    (float): paramètre de lissage pour clDice.
        gamma     (float): exponent de la Focal Loss (>=0).
        alpha     (float|None): facteur d’équilibrage de classe (∈[0,1]) ou None.
    """
    def __init__(
        self,
        w_focal: float = 0.9,
        w_cldice: float = 0.1,
        iter_: int = 3,
        smooth: float = 1.0,
        gamma: float = 2.0,
        alpha: float | None = 0.01,
    ):
        super(SoftclDiceFocalLoss, self).__init__()
        self.w_focal  = w_focal
        self.w_cldice = w_cldice
        self.iter     = iter_
        self.smooth   = smooth
        self.gamma    = gamma
        self.alpha    = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:   (B,1,H,W) bruts non bornés
        targets:  (B,1,H,W) {0,1}
        """
        # 1) probabilités et targets float
        probs     = torch.sigmoid(logits)
        targets_f = targets.float()

        # 2) Binary Focal Loss
        # p_t = p if y=1 else (1-p)
        p_t = probs * targets_f + (1 - probs) * (1 - targets_f)
        # alpha balancing
        if self.alpha is not None:
            alpha_factor = targets_f * self.alpha + (1 - targets_f) * (1 - self.alpha)
        else:
            alpha_factor = 1.0
        # focal term
        focal_term = (1 - p_t).pow(self.gamma)
        # safe log
        log_p_t = torch.log(p_t.clamp(min=1e-6))
        focal_loss = -(alpha_factor * focal_term * log_p_t).mean()

        # 3) Soft clDice Loss
        skel_pred = soft_skel(probs, self.iter)
        skel_true = soft_skel(targets_f, self.iter)

        # aplatissement
        p_flat = skel_pred.view(-1)
        t_flat = skel_true.view(-1)
        y_flat = probs.view(-1)
        g_flat = targets_f.view(-1)

        tprec_num = (p_flat * g_flat).sum()
        tprec     = (tprec_num + self.smooth) / (p_flat.sum() + self.smooth)

        tsens_num = (t_flat * y_flat).sum()
        tsens     = (tsens_num + self.smooth) / (t_flat.sum() + self.smooth)

        cldice_loss = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

        # 4) combinaison finale
        return self.w_focal  * focal_loss + self.w_cldice * cldice_loss


# class SoftclDiceFocalLoss(nn.Module):
#     """
#     Combine la Focal Loss binaire et la Soft clDice pour la segmentation Biapy.

#     Args:
#         w_focal   (float):  poids du terme Focal Loss.
#         w_cldice  (float):  poids du terme Soft clDice.
#         iter_     (int):    nombre itérations pour soft_skel().
#         smooth    (float):  paramètre de lissage pour clDice.
#         gamma     (float):  exponent de la Focal Loss (>=0).
#         alpha     (float):  facteur d'équilibrage alpha de la Focal Loss (∈[0,1]) ou None.
#     """
#     def __init__(
#         self,
#         w_focal: float = 1.0,
#         w_cldice: float = 0.0,
#         iter_: int = 3,
#         smooth: float = 1.0,
#         gamma: float = 5.0,
#         #alpha: float = 0.9,
#     ):
#         super(SoftclDiceFocalLoss, self).__init__()
#         self.w_focal  = w_focal
#         self.w_cldice = w_cldice
#         self.iter     = iter_
#         self.smooth   = smooth
#         self.gamma    = gamma
#         #self.alpha    = alpha

#     def _sigmoid_focal(self, logits: torch.Tensor, targets: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
#         """
#         Implementation de la Focal Loss binaire sur logits (non bornés).
#         """
    
#         bce_loss = logits - logits * targets - F.logsigmoid(logits)
#         # invprobs = log σ(−z) si target=1, log σ(z) si target=0
#         invprobs = F.logsigmoid(-logits * (targets * 2 - 1))
#         #loss = (invprobs * self.gamma).exp() * bce_loss

#         # self.alpha = 
#         # if self.alpha is not None:
#         #     # alpha si target=1, (1−alpha) si target=0
#         #     alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
#         #     loss = alpha_factor * loss
#         modulator = invprobs.mul(self.gamma).exp()
#         alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
#         return alpha_factor * modulator * bce_loss

#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """
#         inputs:  logits bruts du réseau, shape (B,1,H,W) ou (B,1,D,H,W)
#         targets: masques {0,1}, même shape que inputs
#         """
#         # 1) Focal Loss
#         targets_f = targets.float()
#         ratio = 0.007 #à peu près le rapport de pixel objet sur pixel background noir
#         #ratio = targets_f.sum() / targets_f.numel()
#         alpha_dyn = 1.0 - ratio

#         fl = self._sigmoid_focal(inputs, targets_f, alpha_dyn)
#         focal_loss = fl.mean()

#         # 2) Soft clDice
#         probs     = torch.sigmoid(inputs)
#         skel_pred = soft_skel(probs, self.iter)
#         skel_true = soft_skel(targets_f, self.iter)

#         p_flat = skel_pred.view(-1)
#         t_flat = skel_true.view(-1)
#         y_flat = probs.view(-1)
#         g_flat = targets_f.view(-1)

#         tprec_num = (p_flat * g_flat).sum()
#         tprec     = (tprec_num + self.smooth) / (p_flat.sum() + self.smooth)

#         tsens_num = (t_flat * y_flat).sum()
#         tsens     = (tsens_num + self.smooth) / (t_flat.sum() + self.smooth)

#         cldice_loss = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

#         # 3) Combinaison finale
#         loss = self.w_focal * focal_loss + self.w_cldice * cldice_loss
#         return loss

class instance_segmentation_loss:
    def __init__(
        self,
        weights=(1, 0.2),
        out_channels="BC",
        mask_distance_channel=True,
        n_classes=2,
        class_rebalance=False,
        instance_type="regular",
        val_to_ignore: Optional[int] = None,
    ):
        """
        Custom loss that mixed BCE and MSE depending on the ``out_channels`` variable.

        Parameters
        ----------
        weights : 2 float tuple, optional
            Weights to be applied to segmentation (binary and contours) and to distances respectively. E.g. ``(1, 0.2)``,
            ``1`` should be multipled by ``BCE`` for the first two channels and ``0.2`` to ``MSE`` for the last channel.

        out_channels : str, optional
            Channels to operate with.

        mask_distance_channel : bool, optional
            Whether to mask the distance channel to only calculate the loss in those regions where the binary mask
            defined by B channel is present.

        class_rebalance: bool, optional
            Whether to reweight classes (inside loss function) or not.

        instance_type : str, optional
            Type of instances expected. Options are: ["regular", "synapses"]
        """
        assert instance_type in ["regular", "synapses"]

        self.weights = weights
        self.out_channels = out_channels
        self.mask_distance_channel = mask_distance_channel
        self.n_classes = n_classes
        self.d_channel = -2 if n_classes > 2 else -1
        self.class_rebalance = class_rebalance
        self.instance_type = instance_type
        self.val_to_ignore = val_to_ignore
        self.ignore_values = True if val_to_ignore is not None else False
        self.binary_channels_loss = torch.nn.BCEWithLogitsLoss()
        self.distance_channels_loss = torch.nn.L1Loss()
        self.class_channel_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        """
        Calculate instance segmentation loss.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor or list of Tensors
            Predictions.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        if isinstance(y_pred, list):
            _y_pred = y_pred[0]
            _y_pred_class = y_pred[1]
            extra_channels = 1
        else:
            _y_pred = y_pred
            extra_channels = 0

        if self.instance_type == "regular" and "D" in self.out_channels and self.out_channels != "Dv2":
            if self.mask_distance_channel:
                D = _y_pred[:, self.d_channel] * y_true[:, 0]
            else:
                D = _y_pred[:, self.d_channel]

        loss = 0
        if self.instance_type == "regular":
            if self.out_channels == "BC":
                assert (
                    y_true.shape[1] == 2 + extra_channels
                ), f"Seems that the GT loaded doesn't have 2 channels as expected in BC. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    if self.ignore_values:
                        B_weight_mask = B_weight_mask * (y_true[:, 0] != self.val_to_ignore)
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    if self.ignore_values:
                        C_weight_mask = C_weight_mask * (y_true[:, 1] != self.val_to_ignore)
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                else:
                    if self.ignore_values:
                        B_weight_mask = torch.ones((y_true[:, 0].shape)) * (y_true[:, 0] != self.val_to_ignore)
                        B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                        C_weight_mask = torch.ones((y_true[:, 1].shape)) * (y_true[:, 1] != self.val_to_ignore)
                        C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                    else:
                        B_binary_channels_loss = self.binary_channels_loss
                        C_binary_channels_loss = self.binary_channels_loss                        

                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0]) + self.weights[
                    1
                ] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
            
            elif self.out_channels == "BCP":
                assert (
                    y_true.shape[1] == 3 + extra_channels
                ), f"Seems that the GT loaded doesn't have 3 channels as expected in BCP. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    if self.ignore_values:
                        B_weight_mask = B_weight_mask * (y_true[:, 0] != self.val_to_ignore)
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)

                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    if self.ignore_values:
                        C_weight_mask = C_weight_mask * (y_true[:, 1] != self.val_to_ignore)
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)

                    P_weight_mask = weight_binary_ratio(y_true[:, 2])
                    if self.ignore_values:
                        P_weight_mask = P_weight_mask * (y_true[:, 2] != self.val_to_ignore)
                    P_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=P_weight_mask)
                else:
                    if self.ignore_values:
                        B_weight_mask = torch.ones((y_true[:, 0].shape)) * (y_true[:, 0] != self.val_to_ignore)
                        B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                        C_weight_mask = torch.ones((y_true[:, 1].shape)) * (y_true[:, 1] != self.val_to_ignore)
                        C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                        P_weight_mask = torch.ones((y_true[:, 2].shape)) * (y_true[:, 2] != self.val_to_ignore)
                        P_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=P_weight_mask)
                    else:
                        B_binary_channels_loss = self.binary_channels_loss
                        C_binary_channels_loss = self.binary_channels_loss  
                        P_binary_channels_loss = self.binary_channels_loss                       

                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0]) \
                    + self.weights[1] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1]) \
                    + self.weights[2] * P_binary_channels_loss(_y_pred[:, 2], y_true[:, 2])    
            elif self.out_channels == "BCM":
                assert (
                    y_true.shape[1] == 3 + extra_channels
                ), f"Seems that the GT loaded doesn't have 3 channels as expected in BCM. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                    M_weight_mask = weight_binary_ratio(y_true[:, 2])
                    M_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=M_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    C_binary_channels_loss = self.binary_channels_loss
                    M_binary_channels_loss = self.binary_channels_loss
                loss = (
                    self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0])
                    + self.weights[1] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
                    + self.weights[2] * M_binary_channels_loss(_y_pred[:, 2], y_true[:, 2])
                )
            elif self.out_channels == "BCD":
                assert (
                    y_true.shape[1] == 3 + extra_channels
                ), f"Seems that the GT loaded doesn't have 3 channels as expected in BCD. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    C_binary_channels_loss = self.binary_channels_loss
                loss = (
                    self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0])
                    + self.weights[1] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
                    + self.weights[2] * self.distance_channels_loss(D, y_true[:, 2])
                )
            elif self.out_channels == "BCDv2":
                assert (
                    y_true.shape[1] == 3 + extra_channels
                ), f"Seems that the GT loaded doesn't have 3 channels as expected in BCDv2. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    C_weight_mask = weight_binary_ratio(y_true[:, 1])
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    C_binary_channels_loss = self.binary_channels_loss
                loss = (
                    self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0])
                    + self.weights[1] * C_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
                    + self.weights[2] * self.distance_channels_loss(D, y_true[:, 2])
                )
            elif self.out_channels in ["BDv2", "BD"]:
                assert (
                    y_true.shape[1] == 2 + extra_channels
                ), f"Seems that the GT loaded doesn't have 2 channels as expected in BD/BDv2. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0]) + self.weights[
                    1
                ] * self.distance_channels_loss(D, y_true[:, 1])
            elif self.out_channels == "BP":
                assert (
                    y_true.shape[1] == 2 + extra_channels
                ), f"Seems that the GT loaded doesn't have 2 channels as expected in BP. GT shape: {y_true.shape}"
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    P_weight_mask = weight_binary_ratio(y_true[:, 1])
                    P_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=P_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    P_binary_channels_loss = self.binary_channels_loss
                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0]) + self.weights[
                    1
                ] * P_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
            elif self.out_channels == "C":
                if self.class_rebalance:
                    C_weight_mask = weight_binary_ratio(y_true)
                    C_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=C_weight_mask)
                else:
                    C_binary_channels_loss = self.binary_channels_loss
                loss = C_binary_channels_loss(_y_pred, y_true)
            elif self.out_channels in ["A"]:
                if self.class_rebalance:
                    A_weight_mask = weight_binary_ratio(y_true)
                    A_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=A_weight_mask)
                else:
                    A_binary_channels_loss = self.binary_channels_loss
                loss = A_binary_channels_loss(_y_pred, y_true)
            # Dv2
            else:
                loss = self.weights[0] * self.distance_channels_loss(_y_pred, y_true)

            if self.n_classes > 2:
                loss += self.weights[-1] * self.class_channel_loss(_y_pred_class, y_true[:, -1].type(torch.long))
        else:
            if self.out_channels == "BF":
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0])
                # Depending the dimensions more or less channels are present (2 for 2D and 3 for 3D)
                for c in range(1, y_true.shape[1]):
                    if self.mask_distance_channel:
                        loss += self.weights[c] * self.distance_channels_loss(
                            _y_pred[:, c] * (y_true[:, c] != 0), y_true[:, c]
                        )
                    else:
                        loss += self.weights[c] * self.distance_channels_loss(_y_pred[:, c], y_true[:, c])
            elif self.out_channels == "B":
                if self.class_rebalance:
                    B_weight_mask = weight_binary_ratio(y_true[:, 0])
                    B_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=B_weight_mask)
                    BB_weight_mask = weight_binary_ratio(y_true[:, 1])
                    BB_binary_channels_loss = torch.nn.BCEWithLogitsLoss(weight=BB_weight_mask)
                else:
                    B_binary_channels_loss = self.binary_channels_loss
                    BB_binary_channels_loss = self.binary_channels_loss
                loss = self.weights[0] * B_binary_channels_loss(_y_pred[:, 0], y_true[:, 0]) + self.weights[
                    1
                ] * BB_binary_channels_loss(_y_pred[:, 1], y_true[:, 1])
        return loss


def detection_metrics(
    true,
    pred,
    true_classes=None,
    pred_classes=None,
    tolerance=10,
    resolution: List[int|float]=[1, 1, 1],
    bbox_to_consider=[],
    verbose=False,
):
    """
    Calculate detection metrics based on

    Parameters
    ----------
    true : List of list
        List containing coordinates of ground truth points. E.g. ``[[5,3,2], [4,6,7]]``.

    pred : 4D Tensor
        List containing coordinates of predicted points. E.g. ``[[5,3,2], [4,6,7]]``.

    true_classes : List of ints, optional
        Classes of each ground truth points.

    pred_classes : List of ints, optional
        Classes of each predicted points.

    tolerance : optional, int
        Maximum distance far away from a GT point to consider a point as a true positive.

    resolution : List of int/float
        Weights to be multiply by each axis. Useful when dealing with anysotropic data to reduce the distance value
        on the axis with less resolution. E.g. ``(1,1,0.5)``.

    bbox_to_consider : List of tuple/list, optional
        To not take into account during metric calculation to those points outside the bounding box defined with
        this variable. Order is: ``[[z_min, z_max], [y_min, y_max], [x_min, x_max]]``. For example, using an image
        of ``10x100x200`` to not take into account points on the first/last slices and with a border of ``15`` pixel
        for ``x`` and ``y`` axes, this variable could be defined as follows: ``[[1, 9], [15, 85], [15, 185]]``.

    verbose : bool, optional
        To print extra information.

    Returns
    -------
    metrics : List of strings
        List containing precision, accuracy and F1 between the predicted points and ground truth.
    """
    if len(bbox_to_consider) > 0:
        assert len(bbox_to_consider) == 3, "'bbox_to_consider' need to be of length 3"
        assert [len(x) == 2 for x in bbox_to_consider], (
            "'bbox_to_consider' needs to be a list of " "two element array/tuple. E.g. [[1,1],[15,100],[10,200]]"
        )
    if true_classes is not None and pred_classes is None:
        raise ValueError("'pred_classes' must be provided when 'true_classes' is set")

    if true_classes is not None and pred_classes is not None:
        if len(true_classes) != len(true):
            raise ValueError("'true' and 'true_classes' length must be the same")
        if len(pred_classes) != len(pred_classes):
            raise ValueError("'pred' and 'pred_classes' length must be the same")
        class_metrics = True
    else:
        class_metrics = False

    _true = np.array(true, dtype=np.float32)
    _pred = np.array(pred, dtype=np.float32)

    TP, FP, FN = 0, 0, 0
    tag = ["FN" for x in _true]
    fp_preds = list(range(1, len(_pred) + 1))
    dis = [-1 for x in _true]
    pred_id_assoc = [-1 for x in _true]

    TP_not_considered = 0
    if len(_true) > 0:
        # Multiply each axis for the its real value
        for i in range(len(resolution)):
            _true[:, i] *= resolution[i]
            _pred[:, i] *= resolution[i]

        # Create cost matrix
        distances = distance_matrix(_pred, _true)
        n_matched = min(len(_true), len(_pred))
        costs = -(distances >= tolerance).astype(float) - distances / (2 * n_matched)
        pred_ind, true_ind = linear_sum_assignment(-costs)

        # Analyse which associations are below the tolerance to consider them TP
        for i in range(len(pred_ind)):
            # Filter out those point outside the defined bounding box
            consider_point = False
            if len(bbox_to_consider) > 0:
                point = true[true_ind[i]]
                if (
                    bbox_to_consider[0][0] <= point[0] <= bbox_to_consider[0][1]
                    and bbox_to_consider[1][0] <= point[1] <= bbox_to_consider[1][1]
                    and bbox_to_consider[2][0] <= point[2] <= bbox_to_consider[2][1]
                ):
                    consider_point = True
            else:
                consider_point = True

            if distances[pred_ind[i], true_ind[i]] < tolerance:
                if consider_point:
                    TP += 1
                    tag[true_ind[i]] = "TP"
                else:
                    tag[true_ind[i]] = "NC"
                    TP_not_considered += 1
                fp_preds.remove(pred_ind[i] + 1)

            dis[true_ind[i]] = distances[pred_ind[i], true_ind[i]]
            pred_id_assoc[true_ind[i]] = pred_ind[i] + 1

        if TP_not_considered > 0:
            print(f"{TP_not_considered} TPs not considered due to filtering")
        FN = len(_true) - TP - TP_not_considered

    # FP filtering
    FP_not_considered = 0
    fp_tags = ["FP" for x in fp_preds]
    if len(bbox_to_consider) > 0:
        for i in range(len(fp_preds)):
            point = pred[fp_preds[i] - 1]
            if not (
                bbox_to_consider[0][0] <= point[0] <= bbox_to_consider[0][1]
                and bbox_to_consider[1][0] <= point[1] <= bbox_to_consider[1][1]
                and bbox_to_consider[2][0] <= point[2] <= bbox_to_consider[2][1]
            ):
                FP_not_considered += 1
                fp_tags[i] = "NC"

        print(f"{FP_not_considered} FPs not considered due to filtering")
    FP = len(fp_preds) - FP_not_considered

    # Create two dataframes with the GT and prediction points association made and another one with the FPs
    df, df_fp = None, None
    if len(_true) > 0:
        _true = np.array(true, dtype=np.float32)
        _pred = np.array(pred, dtype=np.float32)

        # Capture FP coords
        fp_coords = np.zeros((len(fp_preds), _pred.shape[-1]))
        pred_fp_class = [-1] * len(fp_preds)
        for i in range(len(fp_preds)):
            fp_coords[i] = _pred[fp_preds[i] - 1]
            if class_metrics:
                assert pred_classes is not None
                pred_fp_class[i] = int(pred_classes[fp_preds[i] - 1])

        # Capture prediction coords
        pred_coords = np.zeros((len(pred_id_assoc), 3), dtype=np.float32)
        pred_class = [-1] * len(pred_id_assoc)
        if not class_metrics:
            true_classes = [-1] * len(pred_id_assoc)
        for i in range(len(pred_id_assoc)):
            if pred_id_assoc[i] != -1:
                pred_coords[i] = _pred[pred_id_assoc[i] - 1]
                if class_metrics:
                    assert pred_classes is not None
                    pred_class[i] = int(pred_classes[pred_id_assoc[i] - 1])
            else:
                pred_coords[i] = [0, 0, 0]

        df = pd.DataFrame(
            zip(
                list(range(1, len(_true) + 1)),
                pred_id_assoc,
                dis,
                tag,
                _true[..., 0],
                _true[..., 1],
                _true[..., 2],
                true_classes,  # type: ignore
                pred_coords[..., 0],
                pred_coords[..., 1],
                pred_coords[..., 2],
                pred_class,
            ),  # type: ignore
            columns=[
                "gt_id",
                "pred_id",
                "distance",
                "tag",
                "axis-0",
                "axis-1",
                "axis-2",
                "gt_class",
                "pred_axis-0",
                "pred_axis-1",
                "pred_axis-2",
                "pred_class",
            ],
        )
        df_fp = pd.DataFrame(
            zip(
                fp_preds,
                fp_coords[..., 0],
                fp_coords[..., 1],
                fp_coords[..., 2],
                fp_tags,
                pred_fp_class,
            ),
            columns=["pred_id", "axis-0", "axis-1", "axis-2", "tag", "pred_class"],
        )

    try:
        precision = TP / (TP + FP)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    try:
        F1 = 2 * ((precision * recall) / (precision + recall))
    except:
        F1 = 0

    if not class_metrics:
        if df is not None:
            df = df.drop(columns=["gt_class", "pred_class"])
        if df_fp is not None:
            df_fp = df_fp.drop(columns=["pred_class"])
    else:
        if df is not None:
            gt_matched_classes = df["gt_class"].tolist()
            pred_matched_classes = df["pred_class"].tolist()
            TP_classes = len([1 for x, y in zip(gt_matched_classes, pred_matched_classes) if x == y])
            FN_classes = len([1 for x, y in zip(gt_matched_classes, pred_matched_classes) if x != y])
        else:
            TP_classes = 0
            FN_classes = 0

        try:
            precision_classes = TP_classes / (TP_classes + FP)
        except:
            precision_classes = 0
        try:
            recall_classes = TP_classes / (TP_classes + FN_classes)
        except:
            recall_classes = 0
        try:
            F1_classes = 2 * ((precision_classes * recall_classes) / (precision_classes + recall_classes))
        except:
            F1_classes = 0

    if verbose:
        if len(bbox_to_consider) > 0:
            print(
                "Points in ground truth: {} ({} total but {} not considered), Points in prediction: {} "
                "({} total but {} not considered)".format(
                    len(_true),
                    len(true),
                    TP_not_considered,
                    len(_pred),
                    len(pred),
                    FP_not_considered,
                )
            )
        else:
            print("Points in ground truth: {}, Points in prediction: {}".format(len(_true), len(_pred)))
        print("True positives: {}, False positives: {}, False negatives: {}".format(int(TP), int(FP), int(FN)))
        if class_metrics:
            print("True positives (class): {}, False negatives (class): {}".format(int(TP_classes), int(FN_classes)))

    if not class_metrics:
        r_dict = {
            "Precision": precision,
            "Recall": recall,
            "F1": F1,
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
        }
    else:
        r_dict = {
            "Precision": precision,
            "Recall": recall,
            "F1": F1,
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "Precision (class)": precision_classes,
            "Recall (class)": recall_classes,
            "F1 (class)": F1_classes,
            "TP (class)": int(TP_classes),
            "FN (class)": int(FN_classes),
        }
    return r_dict, df, df_fp


class SSIM_loss(torch.nn.Module):
    def __init__(self, data_range, device):
        super(SSIM_loss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device, non_blocking=True)

    def forward(self, input, target):
        return 1 - self.ssim(input, target)


class W_MAE_SSIM_loss(torch.nn.Module):
    def __init__(self, data_range, device, w_mae=0.5, w_ssim=0.5):
        super(W_MAE_SSIM_loss, self).__init__()
        self.w_mae = w_mae
        self.w_ssim = w_ssim
        self.mse = torch.nn.L1Loss().to(device, non_blocking=True)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device, non_blocking=True)

    def forward(self, input, target):
        return (self.mse(input, target) * self.w_mae) + ((1 - self.ssim(input, target)) * self.w_ssim)


class W_MSE_SSIM_loss(torch.nn.Module):
    def __init__(self, data_range, device, w_mse=0.5, w_ssim=0.5):
        super(W_MSE_SSIM_loss, self).__init__()
        self.w_mse = w_mse
        self.w_ssim = w_ssim
        self.mse = torch.nn.MSELoss().to(device, non_blocking=True)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device, non_blocking=True)

    def forward(self, input, target):
        return (self.mse(input, target) * self.w_mse) + ((1 - self.ssim(input, target)) * self.w_ssim)


def n2v_loss_mse(y_pred, y_true):
    target = y_true[:, 0].squeeze()
    mask = y_true[:, 1].squeeze()
    loss = torch.sum(torch.square(target - y_pred.squeeze() * mask)) / torch.sum(mask)
    return loss


class SSIM_wrapper:
    def __init__(self):
        """
        Wrapper to SSIM loss function.
        """
        self.loss = SSIM(data_range=1, size_average=True, channel=1)

    def __call__(self, y_pred, y_true):
        """
        Calculate instance segmentation loss.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth masks.

        y_pred : torch.Tensor or list of Tensors
            Predictions.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        return 1 - self.loss(y_pred, y_true)
