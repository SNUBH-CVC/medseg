# https://github.com/jocpae/clDice

import torch
import torch.nn as nn


class clDiceLoss(nn.Module):

    def forward(self, pred, y):
        """
        Parameters
        ----------
        pred: tuple
            (pred_mask, pred_skel)
            mask prediction (batch_size, channels, dim_image)
            skeleton prediction (batch_size, channels, dim_image)
        y: tuple
            (y_mask, y_skel)
            mask ground-truth (batch_size, channels, dim_image)
            skeleton ground-truth (batch_size, channels, dim_image)
        """
        pred_mask, pred_skel = pred
        y_mask, y_skel = y
        epsilon = 1e-5
        dim = list(range(1, pred_mask.dim()))
        t_prec = (torch.sum(pred_skel * y_mask, dim=dim) + epsilon) / (
            torch.sum(pred_skel, dim=dim) + epsilon
        )
        t_sens = (torch.sum(y_skel * pred_mask, dim=dim) + epsilon) / (
            torch.sum(y_skel, dim=dim) + epsilon
        )
        cl_dice = torch.mean(1.0 - (2.0 * t_prec * t_sens) / (t_prec + t_sens))
        return cl_dice
