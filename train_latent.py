import os
import sys
import math
import click
import dnnlib
import numpy as np
from PIL import Image
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn as nn
import legacy
import pickle
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
try:
    import wandb
except ImportError:
    wandb = None
import pathlib 
from vgg import Vgg16

from training.networks_encoder import GradualStyleEncoder

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def mse_loss(x, y):
    """
    retain the mse for each instance in a batch
    """
    mse = torch.square(x-y).mean(dim=[i for i in range(1, len(x.shape))])
    return mse


@click.command()
@click.option('--data', help='Input data of textures for inversion', required=True)
@click.option('--g_path', help='Generator network pickle filename', required=True)
@click.option('--e_path', help='Encoder network pickle filename', default=None, type=str)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--resolution', help='Input spatial size to encoder network', metavar='INT', type=click.IntRange(min=0), default=256, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--lr', help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=0.01, show_default=True)
@click.option('--iters', help='Number of training iterations', metavar='INT', type=click.IntRange(min=0), default=400, show_default=True)
@click.option('--batch_size', help='Batch size', metavar='INT', type=click.IntRange(min=0), default=1, show_default=True)
@click.option('--latent_dim', help='Latent dimension', metavar='INT', type=click.IntRange(min=0), default=512, show_default=True)
@click.option('--save_img_every', help='Save images every n iterations', metavar='INT', type=click.IntRange(min=0), default=100, show_default=True)
@click.option('--w_mean', help='Use mean w vector as initialization', type=bool, default=False, show_default=True)
@click.option('--wspace', help='Use w-space or z-space', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--content_weight', help='Loss weight for content loss', metavar='FLOAT', type=float, default=0, show_default=True)
@click.option('--style_weight', help='Loss weight for style loss', metavar='FLOAT', type=float, default=1e5, show_default=True)
@click.option('--pix_weight', help='Loss weight for pixel-wise loss', metavar='FLOAT', type=float, default=0, show_default=True)
def main(
    data: str,
    g_path: str,
    e_path: str,
    seed: int,
    resolution: int,
    outdir: str,
    lr: float,
    iters: int,
    batch_size: int,
    latent_dim: int,
    save_img_every: int,
    w_mean: bool, 
    wspace: bool,
    content_weight: float,
    style_weight: float,
    pix_weight: float,
    ):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    seed_name = "seed%05d"%(seed)

    print('Loading generator network pickle from "%s"...' % g_path)
    with open(g_path, 'rb') as f:
        generator = pickle.load(f)['G_ema'].cuda()
    ckpt_name = os.path.splitext(os.path.basename(g_path.strip("/")))[0]
    num_ws = 16

    encoder = None
    if e_path is not None:
        encoder = GradualStyleEncoder(50, size=resolution).cuda()
        ckpt = torch.load(e_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(ckpt["e"], strict=False) if "e" in ckpt else None
        encoder.eval()

    vgg = Vgg16().to("cuda")
    vgg.eval()    

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")     
    outdir = os.path.join(outdir, "optimization_based_wspace_%r"%wspace, ckpt_name, seed_name, timestr)
    genImageDir = os.path.join(outdir, "gen_images")
    realImageDir = os.path.join(outdir, "real_images")
    latentDir = os.path.join(outdir, "latents")
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(genImageDir):
        os.makedirs(genImageDir, exist_ok=True)
    if not os.path.exists(realImageDir):
        os.makedirs(realImageDir, exist_ok=True)
    if not os.path.exists(latentDir):
        os.makedirs(latentDir, exist_ok=True)

    if w_mean is True:
        print("Computing mean w vector")
        with torch.no_grad():
            latent = torch.randn(100000, latent_dim, device="cuda", requires_grad=False)
            latent = generator.mapping(latent, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)[:,0,:]
            w_mean = torch.mean(latent, dim=0, keepdim=True)

    transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    data = pathlib.Path(data)
    imgs = data.rglob("*.[pj][np]g")
    pbar = tqdm(imgs)
    for img in pbar:
        img_name = os.path.splitext(os.path.basename(str(img).strip("/")))[0]
        real_image = transform(Image.open(img)).to("cuda").unsqueeze(0)
        with torch.no_grad():
            features_y = vgg(real_image)

        latent = torch.randn(batch_size, latent_dim, device="cuda", requires_grad=True)
        if wspace:
            with torch.no_grad():
                if encoder is None:
                    if w_mean is True:
                        latent = w_mean.clone()
                    else:
                        latent = generator.mapping(latent, None, truncation_psi=1, truncation_cutoff=None, update_emas=False).detach()
                        latent = latent[:,0,:]
                else:
                    latent = encoder(real_image*2-1).mean(dim=1).detach().repeat(batch_size, 1)
        latent.requires_grad = True

        optimizer = optim.Adam([latent], lr=lr)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)

        for idx in range(iters+1):
            optimizer.zero_grad()

            w = None
            if wspace:
                w = latent.unsqueeze(1).repeat(1, num_ws ,1)
            else:
                w = generator.mapping(latent, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
            gen_image = 0.5*(1+torch.clamp(generator.synthesis(w, update_emas=False, noise_mode='const'), -1, 1))
            features_x = vgg(gen_image)

            content_loss, style_loss, pixel_loss, channel_mean_loss = 0, 0, 0, 0
            if content_weight != 0:
                content_loss_12 = mse_loss(features_y.conv1_2, features_x.conv1_2)
                content_loss_22 = mse_loss(features_y.conv2_2, features_x.conv2_2)
                content_loss_33 = mse_loss(features_y.conv3_3, features_x.conv3_3)
                content_loss_43 = mse_loss(features_y.conv4_3, features_x.conv4_3)
                content_loss = content_loss_12 + content_loss_22 + content_loss_33 + content_loss_43 
            if style_weight != 0:
                style_loss_12 = mse_loss(gram_matrix(features_y.conv1_2.repeat(batch_size, 1, 1, 1)), gram_matrix(features_x.conv1_2))
                style_loss_22 = mse_loss(gram_matrix(features_y.conv2_2.repeat(batch_size, 1, 1, 1)), gram_matrix(features_x.conv2_2))
                style_loss_33 = mse_loss(gram_matrix(features_y.conv3_3.repeat(batch_size, 1, 1, 1)), gram_matrix(features_x.conv3_3))
                style_loss_43 = mse_loss(gram_matrix(features_y.conv4_3.repeat(batch_size, 1, 1, 1)), gram_matrix(features_x.conv4_3))
                style_loss =  style_loss_12 + style_loss_22 + style_loss_33 + style_loss_43
            if pix_weight != 0:
                pixel_loss = mse_loss(gen_image, real_image)            
            total_loss = content_weight*content_loss + pix_weight*pixel_loss + style_weight*style_loss
            total_loss_sum = torch.mean(total_loss)
            total_loss_sum.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description("loss = %.5f"%total_loss_sum.detach())

            if idx == iters:
                out_gen_img_filename = os.path.join(genImageDir, img_name+".png")
                out_real_img_filename = os.path.join(realImageDir, img_name+".png")
                out_latent_filename = os.path.join(latentDir, img_name+".pt")
                save_image(gen_image[0].clone(), out_gen_img_filename)
                save_image(real_image[0].clone(), out_real_img_filename)
                torch.save(latent, out_latent_filename)               

if __name__ == "__main__":
    main()