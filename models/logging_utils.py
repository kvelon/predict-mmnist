import torchvision
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def make_plot(ctx, tgt, pred, epoch=999, cmap='gray'):
    num_ctx_frames= ctx.shape[1]
    num_tgt_frames = tgt.shape[1]

    def show_frames(frames, ax, row_label=None):
        for i, frame in enumerate(frames):
            if cmap is not None:
                ax[i].imshow(frame, cmap)
            else:
                ax[i].imshow(frame)
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        if row_label is not None:
            ax[0].set_ylabel(row_label)

    ctx_frames = ctx.squeeze().cpu().numpy()
    tgt_frames = tgt.squeeze().cpu().numpy()
    pred_frames = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(3, max(num_ctx_frames, num_tgt_frames),
                       figsize = (9, 5))
    fig.suptitle(f"EPOCH {epoch}", y=0.93)
    show_frames(ctx_frames, ax[0], "Context")
    show_frames(tgt_frames, ax[1], "Target")
    show_frames(pred_frames, ax[2], "Prediction")

    return fig

def make_rgb_plot(ctx, tgt, pred, epoch=999):
    num_ctx_frames= ctx.shape[1]
    num_tgt_frames = tgt.shape[1]

    def show_frames(frames, ax, row_label=None):
        for i, frame in enumerate(frames):
            ax[i].imshow(frame)
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        if row_label is not None:
            ax[0].set_ylabel(row_label)

    ctx_frames = ctx.squeeze().permute(1, 2, 3, 0).cpu().numpy()
    tgt_frames = tgt.squeeze().permute(1, 2, 3, 0).cpu().numpy()
    pred_frames = pred.squeeze().permute(1, 2, 3, 0).cpu().numpy()

    fig, ax = plt.subplots(3, max(num_ctx_frames, num_tgt_frames),
                       figsize = (9, 5))
    fig.suptitle(f"EPOCH {epoch}", y=0.93)
    show_frames(ctx_frames, ax[0], "Context")
    show_frames(tgt_frames, ax[1], "Target")
    show_frames(pred_frames, ax[2], "Prediction")

    return fig

def fig2image(fig):
    # fig.canvas.draw()
    # buf = fig.canvas.tostring_rgb()
    # ncols, nrows = fig.canvas.get_width_height()
    # shp = (nrows, ncols, 3)
    # arr = np.frombuffer(buf, dtype=np.uint8).reshape(shp)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    img = Image.open(buf)
    img = torchvision.transforms.ToTensor()(img)
    return img

def make_ten_frame_plot(ctx, tgt, pred, epoch=999):
    num_ctx_frames= ctx.shape[1]
    num_tgt_frames = tgt.shape[1]

    def show_frames(frames, ax, row_label=None):
        for i, frame in enumerate(frames):
            ax[i+num_ctx_frames].imshow(frame)
            ax[i+num_ctx_frames].set_xticks([])
            ax[i+num_ctx_frames].set_yticks([])

        if row_label is not None:
            ax[0].set_ylabel(row_label)

    ctx_frames = ctx.permute(1, 2, 3, 0).cpu().numpy()
    tgt_frames = tgt.squeeze().permute(1, 2, 3, 0).cpu().numpy()
    pred_frames = pred.squeeze().permute(1, 2, 3, 0).cpu().numpy()

    fig, ax = plt.subplots(2, 10,
                       figsize = (10, 3))
    fig.suptitle(f"EPOCH {epoch}", y=0.93)

    for i, frame in enumerate(ctx_frames):
        ax[0][i].imshow(frame)
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        
        ax[1][i].imshow(frame)
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
    
    ax[0][0].set_ylabel("Ground truth")
    ax[1][0].set_ylabel("Prediction")

    show_frames(tgt_frames, ax[0])
    show_frames(pred_frames, ax[1])

    return fig

def make_plot_image(ctx, tgt, pred, epoch, cmap='gray'):
    if ctx.shape[0] == 1:
        return fig2image(make_plot(ctx, tgt, pred, epoch, cmap='gray'))
    else:
        if ctx.shape[1] < 5:
            return fig2image(make_ten_frame_plot(ctx, tgt, pred, epoch))
        else:
            return fig2image(make_rgb_plot(ctx, tgt, pred, epoch))