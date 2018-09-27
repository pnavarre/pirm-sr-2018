#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pablo Navarrete Michelini
"""

import os
from os.path import basename
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms

from models import G_MGBP


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [
        '.png', '.tif', '.jpg', '.jpeg', '.bmp', '.pgm'
    ])


if __name__ == '__main__':
    out_dir = 'output_images/'

    gpu = 0
    # Region 1
    # noise_W, model_file = 0.0632, 'BOE-R1_CH3_BIC_FE48_LEAK0.0_TR0_BN0_MX0_Analysis#L4#K3#D1_Upscaling#L4#M3#K3#D1_Downscaling#L4#M0#K3#D1_Synthesis#L4#K3#D1_v3_ms.model'
    # Region 2
    # noise_W, model_file = 0.275, 'BOE-R2_CH3_BIC_FE48_LEAK0.0_TR0_BN0_MX0_Analysis#L4#K3#D1_Upscaling#L4#M3#K3#D1_Downscaling#L4#M0#K3#D1_Synthesis#L4#K3#D1_v3_ms.model'
    # Region 3
    noise_W, model_file = 1., 'BOE-R3_CH3_BIC_FE48_LEAK0.0_TR0_BN0_MX0_Analysis#L4#K3#D1_Upscaling#L4#M3#K3#D1_Downscaling#L4#M0#K3#D1_Synthesis#L4#K3#D1_v3_ms.model'

    target_list = [str(f) for f in Path(
        'input_images'
    ).iterdir() if is_image_file(str(f))]

    PIL_to_Tensor = transforms.ToTensor()
    torch.backends.cudnn.benchmark = True
    with torch.cuda.device(gpu):
        with torch.no_grad():
            print('\n- Load model')
            model = G_MGBP(
                'CH' + '.'.join(model_file.split('_CH')[1].split('.')[:-1]),
                name='[gpu%d] G-MGBP' % gpu,
                vlevel=0
            )
            model.set_factor([4, 4])
            model.set_mu(2)
            model.load_state_dict(torch.load(
                model_file, map_location=lambda storage, loc: storage
            ))
            model.train(False)
            model.cuda()

            print('\n- Testing', flush=True)
            print('    ----------------------', flush=True)

            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            Path(out_dir).mkdir(parents=True)

            for target_file in tqdm(target_list):
                key = basename(target_file)

                # Load
                net_input_tensor = PIL_to_Tensor(
                    Image.open(target_file).convert('RGB')
                ).unsqueeze(0).cuda(gpu)

                # Run
                net_output_rgb = model(
                    net_input_tensor,
                    noise_amp=torch.tensor(noise_W).cuda(gpu).float(),
                    pad=True
                ).data[0].clamp(0, 1.).permute(1, 2, 0).cpu().numpy()*255.

                # Save
                out_filename = out_dir + key
                Image.fromarray(
                    np.uint8(np.round(net_output_rgb))
                ).save(out_filename)
