import os
import sys
import math
import click
import dnnlib
import numpy as np
from PIL import Image
import torch
from torch import optim
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import legacy
import pickle
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
try:
    import wandb
except ImportError:
    wandb = None
import pathlib 
from training.networks_encoder import GradualStyleEncoder

@click.command()
@click.option('--data', help='Input data', required=True)
@click.option('--g_path', help='Generator network pickle filename', required=True)
@click.option('--e_path', help='Encoder network pickle filename', required=True)
@click.option('--resolution', help='Input spatial size to encoder network', metavar='INT', type=click.IntRange(min=0), default=256, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--num_layers', help='Number of layers of encoder network', metavar='INT', type=click.IntRange(min=0), default=50, show_default=True)
@click.option('--latent_dim', help='Latent dimension', metavar='INT', type=click.IntRange(min=0), default=512, show_default=True)
@click.option('--wspace', help='Use w-space or z-space', metavar='BOOL', type=bool, default=True, show_default=True)
def main(
    data: str,
    g_path: str,
    e_path: str,
    resolution: int,
    outdir: str,
    num_layers: int,
    latent_dim: int,
    wspace: bool,
    ):
    print('Loading generator network pickle from "%s"...' % g_path)
    with open(g_path, 'rb') as f:
        generator = pickle.load(f)['G_ema'].cuda()
        generator.eval()
    num_ws = generator.synthesis.num_ws
    g_ckpt_name = os.path.splitext(os.path.basename(g_path.strip("/")))[0]     

    encoder = GradualStyleEncoder(num_layers, size=resolution).cuda()
    ckpt = torch.load(e_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(ckpt["e"], strict=False) if "e" in ckpt else None
    encoder.eval()
    e_ckpt_name = os.path.splitext(os.path.basename(e_path.strip("/")))[0] 
 
    outdir = os.path.join(outdir, "texture_analysis", "g_%s_e_%s_wspace_%r"%(g_ckpt_name, e_ckpt_name, wspace))
    genImageDir = os.path.join(outdir, "gen_images")
    realImageDir = os.path.join(outdir, "real_images")
    
    if not os.path.exists(genImageDir):
        os.makedirs(genImageDir, exist_ok=True)        
    if not os.path.exists(realImageDir):
        os.makedirs(realImageDir, exist_ok=True)        

    onecrop = transforms.Compose([transforms.CenterCrop(resolution), transforms.ToTensor()])

    data = pathlib.Path(data)
    imgs = data.rglob("*.[pj][np]g")
    pbar = tqdm(imgs)

    for img in pbar:
        img_name = os.path.splitext(os.path.basename(str(img).strip("/")))[0]
        image = Image.open(img).convert("RGB")
        with torch.no_grad():
            real_image = onecrop(image).to("cuda").unsqueeze(0)
            code = encoder(real_image*2-1).mean(dim=1)       
            if wspace:
                code = code.unsqueeze(1).repeat(1, num_ws, 1)
            else:
                code = generator.mapping(code, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
            gen_image = torch.clamp(generator.synthesis(code, update_emas=False, noise_mode='const'), min=-1, max=1)
            gen_image = 0.5*(1+gen_image)
        out_gen_img_path = os.path.join(genImageDir, img_name+".png")
        out_real_img_path = os.path.join(realImageDir, img_name+".png")
        save_image(gen_image[0], out_gen_img_path)
        save_image(real_image[0], out_real_img_path)

if __name__ == "__main__":
    main()