import os
import sys
import math
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import legacy
import pickle
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
try:
    import wandb
except ImportError:
    wandb = None

from training.networks_encoder import GradualStyleEncoder

@click.command()
@click.option('--g_path', help='Generator network pickle filename', required=True)
@click.option('--e_path', help='Encoder network pickle filename', default=None, type=str)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--resolution', help='Input spatial size to encoder network', metavar='INT', type=click.IntRange(min=0), default=256, show_default=True)
@click.option('--outdir', help='Where to save the model checkpoint', type=str, required=True, metavar='DIR')
@click.option('--elr', help='Encoder learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0001, show_default=True)
@click.option('--num_layers', help='Number of layers of encoder network', metavar='INT', type=click.IntRange(min=0), default=50, show_default=True)
@click.option('--iters', help='Number of training iterations', metavar='INT', type=click.IntRange(min=0), default=500000, show_default=True)
@click.option('--batch_size', help='Batch size', metavar='INT', type=click.IntRange(min=0), default=16, show_default=True)
@click.option('--latent_dim', help='Latent dimension', metavar='INT', type=click.IntRange(min=0), default=512, show_default=True)
@click.option('--save_img_every', help='Save images every n iterations', metavar='INT', type=click.IntRange(min=0), default=500, show_default=True)
@click.option('--save_ckpt_every', help='Save models every n iterations', metavar='INT', type=click.IntRange(min=0), default=50000, show_default=True)
@click.option('--wspace', help='Use w-space or z-space', metavar='BOOL', type=bool, default=True, show_default=True)
def main(
    g_path: str,
    e_path: str,
    seed: int,
    resolution: int,
    outdir: str,
    elr: float,
    num_layers: int,
    iters: int,
    batch_size: int,
    latent_dim: int,
    save_img_every: int,
    save_ckpt_every: int,
    wspace: bool,
    ):
    print('Loading generator network pickle from "%s"...' % g_path)
    with open(g_path, 'rb') as f:
        generator = pickle.load(f)['G_ema'].cuda()
    ckpt_name = os.path.splitext(os.path.basename(g_path.strip("/")))[0]
    seed_name = "seed%05d"%(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)        

    encoder = GradualStyleEncoder(num_layers, size=resolution).cuda()
    e_optim = optim.Adam(encoder.parameters(), lr=elr, betas=(0.9, 0.999))
    if e_path is not None:
        ckpt = torch.load(e_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(ckpt["e"], strict=False) if "e" in ckpt else None
        e_optim.load_state_dict(ckpt["e_optim"]) if "e_optim" in ckpt else None
    scheduler = StepLR(e_optim, step_size=iters//2, gamma=0.2)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")     
    outdir = os.path.join(outdir, "encoder_wspace_%r"%wspace, ckpt_name, seed_name, timestr)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if wandb is not None:
        wandb.init(project="LearningBasedGANInversion")

    pbar = tqdm(range(iters+1), file=sys.stdout) 
    num_ws = generator.synthesis.num_ws
    loss_func = nn.MSELoss()
        
    for idx in pbar:
        c = None
        z_in = torch.randn(batch_size, latent_dim, device="cuda", requires_grad=False)
        w_in = generator.mapping(z_in, c, truncation_psi=1, truncation_cutoff=None, update_emas=False) # (batch, num_ws, latent_dim)
        fake_img = torch.clamp(generator.synthesis(w_in, update_emas=False, noise_mode='const'), min=-1, max=1)
        code_out = encoder(fake_img.detach()).mean(dim=1) # (batch, latent_dim)

        latent_in = w_in[:, 0, :] if wspace else z_in
        loss = loss_func(code_out, latent_in).mean()#loss_func(code_out, latent_in)
        encoder.zero_grad()
        loss.backward()
        e_optim.step()
        scheduler.step()

        if wandb is not None:
            logs = {"EncoderLatentCodeL2Loss":loss.detach()}
            if idx % save_img_every == 0:
                fake_img = (fake_img.detach()+1)*0.5
                fake_img = wandb.Image(fake_img, caption="Generated textures")
                logs["Ground Truth Synthesized Image"] = fake_img
                with torch.no_grad():
                    if wspace:
                        code_out = code_out.unsqueeze(1).repeat(1, num_ws, 1)            
                    else:
                        code_out = generator.mapping(code_out, c, truncation_psi=1, truncation_cutoff=None, update_emas=False) # (batch, num_ws, latent_dim)

                    fake_img_invert = torch.clamp(generator.synthesis(code_out.detach(), update_emas=False, noise_mode='const'), min=-1, max=1)
                    fake_img_invert = (fake_img_invert.detach()+1)*0.5
                    fake_img_invert = wandb.Image(fake_img_invert, caption="Inverted textures")
                    logs["Inverted Synthesized Image"] = fake_img_invert         
            wandb.log(logs)

        if idx % save_ckpt_every == 0:
            torch.save({"e":encoder.state_dict(), "e_optim": e_optim.state_dict()}, f"{outdir}/{str(idx).zfill(10)}.pt")

if __name__ == "__main__":
    main()
