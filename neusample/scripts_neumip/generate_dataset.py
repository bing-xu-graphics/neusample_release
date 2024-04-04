import argparse
import os
from re import A
import sys
import struct
from typing import List, Optional
import numpy as np
from pathlib import Path

p = os.path.abspath('.')
sys.path.insert(1, p)
import training_dataset.generator
import training_dataset.blender_render_dataset

from dataclasses import dataclass, field
from omegaconf import OmegaConf

@dataclass
class BaseInputConfig:
    """Default Config"""
    name: str = 'unnamed'
    dataset_name: str = "data/input_datasets/input.query"
    blend_file: str = ""
    num_images: int = 1
    fixed_camera: Optional[bool] = False
    fixed_light: Optional[bool] = False
    fixed_patch: Optional[bool] = False
    resolution: List[int] = field(default_factory=lambda: [512, 512])

class InputsGenerator:

    def __init__(self, conf) -> None:
        self.fixed_camera = conf.fixed_camera
        self.fixed_light = conf.fixed_light
        self.fixed_patch = conf.fixed_patch
        self.num_images = conf.num_images
        self.resolution = conf.resolution
        self.num_level = int(np.ceil(np.log2(np.max(self.resolution))))
        self.base_radius = None
        self.aux_buffers = training_dataset.generator.Buffers(self.resolution, self.num_images)

    def generate(self, dataset_name):
        out_path = Path(dataset_name)

        if self.base_radius is None:
            self.base_radius = training_dataset.generator.GeneratorHelper.calc_base_radius(self.resolution[0])

        if out_path.exists():
            print("Query file already exists, continuing...")
            return

        if not out_path.parent.exists():
            Path.mkdir(out_path.parent)

        write_path = open(out_path, "wb")

        for idx in range(self.num_images):
            self.aux_buffers.generate_buffer(idx, self.num_level, self.fixed_camera, self.fixed_light, self.fixed_patch)

            data = self.aux_buffers.get_buffers()

            bin = struct.pack(len(data)*"f", *data)
            write_path.write(bin)

        print(write_path)

def get_config(config_filename):
    conf: BaseInputConfig = OmegaConf.structured(BaseInputConfig)
    file_conf = OmegaConf.load(config_filename)
    conf = OmegaConf.merge(conf, file_conf)
    return conf

def main(args):
    if args.config_file == None:
        conf = get_config('config/generate_example.yml')
    else:
        conf = get_config(args.config_file)

    # Allow override of dataset name from cli args
    if args.dataset_name is not None:
        conf.dataset_name = args.name

    inputs_only = args.mode == "inputs"
    if inputs_only:
        assert(conf.dataset_name.endswith(".query"))
    else:
        assert(conf.dataset_name.endswith(".h5"))

    query_file_path = Path(conf.dataset_name)
    if not inputs_only:
        query_file_path = query_file_path.with_suffix('.query')

    ig = InputsGenerator(conf)
    ig.generate(str(query_file_path))

    if not inputs_only:
        assert(conf.blend_file.endswith(".blend"))
        training_dataset.blender_render_dataset.render(conf.blend_file, str(query_file_path), conf.dataset_name, conf.num_images, conf.resolution, ig.base_radius)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=["all", "inputs"], default="all", help="full dataset or inputs only")
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--dataset_name', type=str)

    args = parser.parse_args()

    main(args)