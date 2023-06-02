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

from training.networks_encoder import GradualStyleEncoder

SCALING_FACTOR = 1

class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = torch.nn.Conv2d(3, 64, (3,3), padding=(1,1), padding_mode='reflect')
        self.block1_conv2 = torch.nn.Conv2d(64, 64, (3,3), padding=(1,1), padding_mode='reflect')

        self.block2_conv1 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1), padding_mode='reflect')
        self.block2_conv2 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1), padding_mode='reflect')

        self.block3_conv1 = torch.nn.Conv2d(128, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv2 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv3 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv4 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')

        self.block4_conv1 = torch.nn.Conv2d(256, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv2 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv3 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv4 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsampling = torch.nn.AvgPool2d((2,2))

    def forward(self, image):
        
        # RGB to BGR
        image = image[:, [2,1,0], :, :]

        # [0, 1] --> [0, 255]
        image = 255 * image

        # remove average color
        image[:,0,:,:] -= 103.939
        image[:,1,:,:] -= 116.779
        image[:,2,:,:] -= 123.68

        # block1
        block1_conv1 = self.relu(self.block1_conv1(image))
        block1_conv2 = self.relu(self.block1_conv2(block1_conv1))
        block1_pool = self.downsampling(block1_conv2)

        # block2
        block2_conv1 = self.relu(self.block2_conv1(block1_pool))
        block2_conv2 = self.relu(self.block2_conv2(block2_conv1))
        block2_pool = self.downsampling(block2_conv2)

        # block3
        block3_conv1 = self.relu(self.block3_conv1(block2_pool))
        block3_conv2 = self.relu(self.block3_conv2(block3_conv1))
        block3_conv3 = self.relu(self.block3_conv3(block3_conv2))
        block3_conv4 = self.relu(self.block3_conv4(block3_conv3))
        block3_pool = self.downsampling(block3_conv4)

        # block4
        block4_conv1 = self.relu(self.block4_conv1(block3_pool))
        block4_conv2 = self.relu(self.block4_conv2(block4_conv1))
        block4_conv3 = self.relu(self.block4_conv3(block4_conv2))
        block4_conv4 = self.relu(self.block4_conv4(block4_conv3))

        return [block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1, block3_conv2, block3_conv3, block3_conv4, block4_conv1, block4_conv2, block4_conv3, block4_conv4]

def slicing_loss(list_activations_generated, list_activations_example):
    
    # generate VGG19 activations
    # list_activations_generated = vgg(image_generated)
    # list_activations_example   = vgg(image_example)
    
    # iterate over layers
    loss = 0
    for l in range(len(list_activations_example)):
        # get dimensions
        b = list_activations_example[l].shape[0]
        dim = list_activations_example[l].shape[1]
        n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]
        # linearize layer activations and duplicate example activations according to scaling factor
        activations_example = list_activations_example[l].view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
        activations_generated = list_activations_generated[l].view(b, dim, n*SCALING_FACTOR*SCALING_FACTOR)
        # sample random directions
        Ndirection = dim
        directions = torch.randn(Ndirection, dim).to(torch.device("cuda:0"))
        directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
        # project activations over random directions
        projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
        projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions)
        # sort the projections
        sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
        sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]
        # L2 over sorted lists
        loss += torch.mean( (sorted_activations_example-sorted_activations_generated)**2 ) 
    return loss


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
@click.option('--save_img_every', help='Save images every n iterations', metavar='INT', type=click.IntRange(min=0), default=1, show_default=True)
@click.option('--w_mean', help='Use mean w vector as initialization', type=bool, default=False, show_default=True)
@click.option('--wspace', help='Use w-space or z-space', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--content_weight', help='Loss weight for content loss', metavar='FLOAT', type=float, default=0, show_default=True)
@click.option('--channel_mean_weight', help='Loss weight for channel-wise mean loss', metavar='FLOAT', type=float, default=0, show_default=True)
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
    channel_mean_weight: float,
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

    vgg = VGG19().to(torch.device("cuda:0"))
    vgg.load_state_dict(torch.load("vgg19.pth")) 


    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")     
    outdir = os.path.join(outdir, "optimization_based_wspace_%r"%wspace, ckpt_name, seed_name, timestr)
    genImageDir = os.path.join(outdir, "gen_images")
    realImageDir = os.path.join(outdir, "real_images")
    latentDir = os.path.join(outdir, "latents")
    logDirSorted = os.path.join(outdir, "logsSorted")
    logDirUnsorted = os.path.join(outdir, "logsUnsorted")
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(genImageDir):
        os.makedirs(genImageDir, exist_ok=True)
    if not os.path.exists(realImageDir):
        os.makedirs(realImageDir, exist_ok=True)
    if not os.path.exists(latentDir):
        os.makedirs(latentDir, exist_ok=True)
    if not os.path.exists(logDirSorted):
        os.makedirs(logDirSorted, exist_ok=True)
    if not os.path.exists(logDirUnsorted):
        os.makedirs(logDirUnsorted, exist_ok=True)

    if w_mean is True:
        print("Computing mean w vector")
        with torch.no_grad():
            latent = torch.randn(100000, latent_dim, device="cuda", requires_grad=False)
            latent = generator.mapping(latent, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)[:,0,:]
            w_mean = torch.mean(latent, dim=0, keepdim=True)


    transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    data = pathlib.Path(data)
    imgs = data.rglob("*.png")
    # imgs = data.rglob("*.tiff")
    pbar = tqdm(imgs)
    for img in pbar:
        img_name = os.path.splitext(os.path.basename(str(img).strip("/")))[0]
        real_image = transform(Image.open(img)).to("cuda").unsqueeze(0)
        latent = torch.randn(batch_size, latent_dim, device="cuda", requires_grad=True)
        if wspace:
            with torch.no_grad():
                list_activations_example = vgg(real_image)
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
        for idx in range(1, iters+1):
            optimizer.zero_grad()

            w = None
            if wspace:
                w = latent.unsqueeze(1).repeat(1, num_ws ,1)
            else:
                w = generator.mapping(latent, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
            gen_image = 0.5*(1+torch.clamp(generator.synthesis(w, update_emas=False, noise_mode='const'), -1, 1))
            list_activations_generated = vgg(gen_image)
            loss = slicing_loss(list_activations_generated, list_activations_example)
            loss.backward()      

            optimizer.step()

        out_gen_img_filename = os.path.join(genImageDir, img_name+".png")
        out_real_img_filename = os.path.join(realImageDir, img_name+".png")
        save_image(gen_image[0], out_gen_img_filename)
        save_image(real_image[0], out_real_img_filename)        

if __name__ == "__main__":
    main()