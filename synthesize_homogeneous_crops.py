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

    vgg = VGG19().to(torch.device("cuda"))
    vgg.load_state_dict(torch.load("vgg19.pth")) 

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
            list_activations_example = vgg(image)
            w_encoder_init = encoder(2*image-1).mean(dim=1).detach()

        for j in range(1, samples_per_texture):
            img_filename = os.path.join(img_folder, "%05d_%05d.png"%(i, j))
            
            latent = w_encoder_init.clone() + sigma*torch.randn_like(w_encoder_init)
            latent.requires_grad = True
            optimizer = optim.Adam([latent], lr=0.01)
            for idx in range(20+1):
                optimizer.zero_grad()
                w = latent.unsqueeze(1).repeat(1, num_ws ,1)
                gen_image = 0.5*(1+torch.clamp(generator.synthesis(w, update_emas=False, noise_mode='const'), -1, 1))
                list_activations_generated = vgg(gen_image)
                loss = slicing_loss(list_activations_generated, list_activations_example)
                loss.backward()  
                optimizer.step()  
            save_image(gen_image.clone(), img_filename)

if __name__ == "__main__":
    main()


