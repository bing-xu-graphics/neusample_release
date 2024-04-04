"""Datasets I/O and management"""
import h5py
import os
import numpy as np

from utils import exr

import torch as th
from torch.utils.data import Dataset

class NeuMIPv1Dataset(Dataset):
    """Dataset that contains inputs and output of NeuMIP (v1)"""

    # Loads all the training data into memory.
    # Fine for now, but might need to revisit in case it gets too large
    # to fit in GPU memory.
    def __init__(self, datafile, device='cuda:0') -> None:
        super().__init__()

        assert(os.path.exists(datafile))

        self.datafile = datafile

        dataset = h5py.File(self.datafile, 'r')
        tmp_names = list(dataset.values())
        print(tmp_names)

        self.data = dataset
        self.num_elements = dataset["ground_color"].shape[0]

        if "resolution" in dataset.attrs:
            self.resolution = dataset.attrs["resolution"]
        else:
            print("settting resolution to [512,512]")
            self.resolution = [512, 512]

        if "base_radius" in dataset.attrs:
            self.base_radius = dataset.attrs["base_radius"]
        else:
            self.base_radius = 1.0 / self.resolution[1] / 2.0

        # self.num_slices = int(self.num_elements / float(self.resolution[0] * self.resolution[1]))
        self.num_slices = int(self.num_elements)# / float(self.resolution[0] * self.resolution[1]))


        def to_float(x):
            return th.tensor(x[()], dtype=th.float32)

        self.color = to_float(dataset["ground_color"]).to(device)
        # self.light_dir = to_float(dataset["ground_light_dir"]).to(device)
        self.light_dir = to_float(dataset["ground_light"]).to(device)

        self.camera_dir = to_float(dataset["ground_camera_dir"]).to(device)
        self.camera_query_radius = to_float(dataset["ground_camera_query_radius"]).to(device)
        self.location = to_float(dataset["ground_camera_target_loc"]).to(device)

        self.color = self.color.reshape(self.num_slices, -1, self.color.shape[-1])
        self.light_dir = self.light_dir.reshape(self.num_slices, -1, self.light_dir.shape[-1])
        self.camera_dir = self.camera_dir.reshape(self.num_slices, -1, self.camera_dir.shape[-1])
        self.camera_query_radius = self.camera_query_radius.reshape(self.num_slices, -1, self.camera_query_radius.shape[-1])
        self.location = self.location.reshape(self.num_slices, -1, self.location.shape[-1])

        # self.dump_h5()


    def __len__(self):
        return self.num_slices

    def __getitem__(self, index):
        camera_dir = self.camera_dir[index]

        light_dir = self.light_dir[index]

        color = self.color[index]

        location = self.location[index]

        query_rad = self.camera_query_radius[index]
        return camera_dir, light_dir, color, location, query_rad

    def dump_h5(self, outdir="", frames=1):
        def to_float(x):
            return np.array(x[()], dtype=np.float32)
        camera_dir = to_float(self.data["ground_camera_dir"])
        # light_dir = to_float(self.data["ground_light_dir"])
        light_dir = to_float(self.data["ground_light"])

        color = to_float(self.data["ground_color"])
        location = to_float(self.data["ground_camera_target_loc"])

        if outdir is None:
            outdir = os.curdir
        outpath = os.path.abspath(outdir)
        if not os.path.exists(outpath):
            os.mkdir(outpath)

        for i in range(frames):
            color = self.color.reshape(self.num_slices, self.resolution[0], self.resolution[1], self.color.shape[-1])[i]
            light_dir = self.light_dir.reshape(self.num_slices, self.resolution[0], self.resolution[1], self.light_dir.shape[-1])[i]
            camera_dir = self.camera_dir.reshape(self.num_slices, self.resolution[0], self.resolution[1], self.camera_dir.shape[-1])[i]
            location = self.location.reshape(self.num_slices, self.resolution[0], self.resolution[1], self.location.shape[-1])[i]

            camera_dir = np.dstack([camera_dir.cpu(), np.zeros(self.resolution)])
            light_dir = np.dstack([light_dir.cpu(), np.zeros(self.resolution)])
            location = np.dstack([location.cpu(), np.zeros(self.resolution)])
            color = np.array(color.cpu())

            exr.write16(camera_dir, os.path.join(outpath, "camera_dir" + str(i) + ".exr"))
            exr.write16(light_dir, os.path.join(outpath, "light_dir" + str(i) + ".exr"))
            exr.write16(color, os.path.join(outpath, "color" + str(i) + ".exr"))
            exr.write16(location, os.path.join(outpath, "loc" + str(i) + ".exr"))

        return