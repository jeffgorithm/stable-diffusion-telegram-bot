import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

import logging

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
    )

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

class StableDiffusion():
    def __init__(self, config, ckpt, sampler='ddim'):
        # Model
        self.model = None
        self.load_model_from_config(
            config=config,
            ckpt=ckpt
            )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        # Sampler
        self.sampler = None
        self.load_sampler(option=sampler)

        # Configurations
        self.steps = 50 # steps
        self.n_iter = 2 # number of iterations
        self.H = 512 # height
        self.W = 512 # width
        self.C = 4 # channel
        self.f = 8 # downsampling factor, most often 8 or 16
        self.n_samples = 2 # how many samples to produce for each given prompt. A.k.a batch size
        self.scale = 9.0 # unconditional guidance scale
        self.ddim_eta = 0.0 # ddim eta (eta=0.0 corresponds to deterministic sampling


    def load_model_from_config(self, config, ckpt, verbose=False):
        config = OmegaConf.load(f"{config}")
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            logger.info(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        self.model = instantiate_from_config(config.model)
        m, u = self.model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            logger.info("missing keys:")
            logger.info(m)
        if len(u) > 0 and verbose:
            logger.info("unexpected keys:")
            logger.info(u)

        self.model.cuda()
        self.model.eval()

    # Function to load sampler
    def load_sampler(self, option):
        if option == 'plms':
            self.sampler = PLMSSampler(self.model)
        elif option == 'dpm':
            self.sampler = DPMSolverSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)

    def generate(self, prompt, start_code=None, precision='autocast'):
        all_samples = []
        count = 0

        # Prepare prompt
        data = [self.n_samples * [prompt]]

        precision_scope = autocast if precision == "autocast" else nullcontext
        with torch.no_grad(), precision_scope('cuda'), self.model.ema_scope():
            for _ in trange(self.n_iter):
                for prompts in tqdm(data):
                    uc = None

                    if self.scale != 1.0:
                        uc = self.model.get_learned_conditioning(self.n_samples * [""])

                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    c = self.model.get_learned_conditioning(prompts)
                    shape = [self.C, self.H // self.f, self.W // self.f]
                    
                    samples, _ = self.sampler.sample(
                        S=self.steps,
                        conditioning=c,
                        batch_size=self.n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=self.scale,
                        unconditional_conditioning=uc,
                        eta=self.ddim_eta,
                        x_T=start_code
                        )

                    x_samples = self.model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save('{}.png'.format(str(count)))
                        count += 1

                    all_samples.append(x_samples)

        return all_samples

    def save_grid(self, all_samples, rows=2):
        # Create grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=rows)

        # Create image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))
        grid.save('./grid.png')

        return grid

if __name__ == "__main__":
    generator = StableDiffusion(
        config='./configs/stable-diffusion/v2-inference.yaml',
        ckpt='./v2-1_512-ema-pruned.ckpt'
    )
    samples = generator.generate(prompt='portrait of a cyberpunk queen trending in artstation hd')
    grid = generator.save_grid(samples)
    