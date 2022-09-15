import torch
import torch.nn as nn
import numpy as np

from skimage.metrics import structural_similarity, peak_signal_noise_ratio


class ImageGradientDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat, alpha=1):
        #n, c, f, h, w = y.shape
        y_hori_grad = torch.abs(torch.diff(y, axis=4))
        yhat_hori_grad = torch.abs(torch.diff(yhat, axis=4))

        y_vert_grad = torch.abs(torch.diff(y, axis=3))
        yhat_vert_grad = torch.abs(torch.diff(yhat, axis=3))

        hori_grad= torch.abs(y_hori_grad - yhat_hori_grad)
        vert_grad = torch.abs(y_vert_grad - yhat_vert_grad)

        gdl = hori_grad.sum() + vert_grad.sum()
        return gdl

class PSNR(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_frames, target_frames):
        N, C, F, H, W = pred_frames.shape
        out = np.zeros((N, F))
        for vid_idx in range(N):
            for frame_idx in range(F):
                psnr = peak_signal_noise_ratio(
                    pred_frames[vid_idx, :, frame_idx].detach().cpu().numpy(),
                    target_frames[vid_idx, :, frame_idx].detach().cpu().numpy(),
                    data_range=1.0
                )
                out[vid_idx, frame_idx] = psnr
        return out.mean()

class SSIM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_frames, target_frames):
        N, C, F, H, W = pred_frames.shape
        out = np.zeros((N, F))
        for vid_idx in range(N):
            for frame_idx in range(F):
                if C == 1:
                    ssim = structural_similarity(
                        pred_frames[vid_idx, :, frame_idx].detach().cpu().numpy().squeeze(),
                        target_frames[vid_idx, :, frame_idx].detach().cpu().numpy().squeeze(),
                        data_range=1.0
                    )
                else:
                    ssim = structural_similarity(
                    pred_frames[vid_idx, :, frame_idx].detach().cpu().numpy(),
                    target_frames[vid_idx, :, frame_idx].detach().cpu().numpy(),
                    data_range=1.0,
                    channel_axis=0
                )
                out[vid_idx, frame_idx] = ssim
        return out.mean()
