import os
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import legacy
import pickle

from tqdm import tqdm

from typing import List, Optional, Tuple, Union

from training.networks_encoder import GradualStyleEncoder
from torchvision.utils import save_image
from vgg import Vgg16

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
@click.option('--g_path', help='generator', required=True)
@click.option('--e_path', help='encoder', default=None )
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--n_textures', help='Number of different textures to be generated', metavar='INT', type=click.IntRange(min=0), default=10000, show_default=True)
@click.option('--samples_per_texture', help='Number of different samples from the same textures', metavar='INT', type=click.IntRange(min=0), default=20, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--sigma', help='Noise to perturb w vector', metavar='FLOAT', type=click.FloatRange(min=0), default=0.1)
@click.option('--style_weight', help='Loss weight for style loss', metavar='FLOAT', type=float, default=1e5, show_default=True)
def main(
    g_path: str,
    e_path: str,
    seed: int,
    n_textures: int,
    samples_per_texture: int,
    outdir: str,
    sigma: float,
    style_weight: float,
    ):
    with open(g_path, 'rb') as f:
        generator = pickle.load(f)['G_ema'].cuda()
    num_ws = generator.synthesis.num_ws

    encoder = GradualStyleEncoder(50, size=256).cuda()
    if e_path is not None:
        ckpt = torch.load(e_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(ckpt["e"], strict=False) if "e" in ckpt else None
    encoder.eval()

    vgg = Vgg16().to("cuda")
    vgg.eval()  

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda"
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, "latents"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "images"), exist_ok=True)

    for i in tqdm(range(n_textures)):       
        with torch.no_grad():    
            z = torch.randn(1, 512, device=device, requires_grad=False)
            z_filename = os.path.join(outdir, "latents", "%05d.pt" % i)
            torch.save(z, z_filename)
            img_folder = os.path.join(outdir, "images", "%05d"%i)
            if not os.path.exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            w = generator.mapping(z, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
            image = 0.5*(1+torch.clamp(generator.synthesis(w, update_emas=False, noise_mode='const'), min=-1, max=1))
            img_filename = os.path.join(img_folder, "%05d_%05d.png"%(i, 0))
            save_image(image.clone(), img_filename)
            features_y = vgg(image)
            w_encoder_init = encoder(2*image-1).mean(dim=1).detach()

        for j in range(1, samples_per_texture):
            img_filename = os.path.join(img_folder, "%05d_%05d.png"%(i, j))
            
            latent = w_encoder_init.clone() + sigma*torch.randn_like(w_encoder_init)
            latent.requires_grad = True
            optimizer = optim.Adam([latent], lr=0.01)
            for idx in range(20+1):
                optimizer.zero_grad()
                w = latent.unsqueeze(1).repeat(1, num_ws ,1)
                image = 0.5*(1+torch.clamp(generator.synthesis(w, update_emas=False, noise_mode='const'), -1, 1))
                features_x = vgg(image)
                style_loss_12 = mse_loss(gram_matrix(features_y.conv1_2), gram_matrix(features_x.conv1_2))
                style_loss_22 = mse_loss(gram_matrix(features_y.conv2_2), gram_matrix(features_x.conv2_2))
                style_loss_33 = mse_loss(gram_matrix(features_y.conv3_3), gram_matrix(features_x.conv3_3))
                style_loss_43 = mse_loss(gram_matrix(features_y.conv4_3), gram_matrix(features_x.conv4_3))
                style_loss =  style_loss_12 + style_loss_22 + style_loss_33 + style_loss_43        
                total_loss = style_weight*style_loss
                total_loss_sum = torch.mean(total_loss)
                total_loss_sum.backward()
                optimizer.step()  
            save_image(image.clone(), img_filename)

if __name__ == "__main__":
    main()


