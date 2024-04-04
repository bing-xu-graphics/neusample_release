"""Image transforms and other utility functions."""
import os
import imageio
import numpy as np
import torch as th
# import pyexr


def psnr(loss, eps=1e-8):
    psnr_ = -10*np.log10(loss.square().mean().numpy() + eps)
    return psnr_

def append_alpha_one(im):
    imshape = im.shape[:2]
    alpha = th.ones(imshape[0], imshape[1], 1)
    return th.cat([im, alpha], dim=-1)

def alpha_multiply(im, background=None):
    alpha = im[:, 3:4]
    im = im[:, :3]
    if background is None:
        return im
    else:
        return im + (1.0-alpha)*background


def tonemap(img, gamma=2.2):
    if type(img) == th.Tensor:
        img = th.clamp(img, min=0.0, max=None)
        img = (img / (1 + img)) ** (1.0 / gamma)
    elif type(img) == np.ndarray:
        img = np.clip(img, a_min=0.0, a_max=None)
        img = (img / (1 + img)) ** (1.0 / gamma)
    else:
        raise RuntimeError("Unknown data type!")

    return img


def imsave(path: str, image: th.Tensor):
    """Save a tensor as an image on disk.
    Args:
        path(str): path to the destination file.
        image(th.Tensor): image tensor with shape [1, c, h, w].
    """

    assert len(image.shape) == 4, "expected 4D tensor"
    assert image.size(0) == 1, "expected batch size to be 1"

    image = image.squeeze(0).permute(1, 2, 0)

    ext = os.path.splitext(path)[-1]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if ext in [".png", ".jpg", ".jpeg"]:
        image = th.clamp(image, 0, 1)
        image = image.detach().cpu().numpy()
        image = (image*255).astype(np.uint8)
        imageio.imsave(path, image)
    elif ext == ".exr":
        image = th.clamp(image, 0)
        image = image.detach().cpu().numpy()
        # pyexr.write(path, image)
    else:
        raise ValueError(f"Unknown image output format {ext}")
