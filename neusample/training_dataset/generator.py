import torch
import numpy as np
import utils
import math

class Buffers:
    def __init__(self, resolution, num_slices) -> None:
        self.camera_origin = None
        self.camera_target_loc = None
        self.camera_dir = None
        self.camera_query_radius = None
        self.base_radius = None
        self.num_slices = num_slices
        self.generator = GeneratorHelper(resolution, self.num_slices)

    def get_buffers(self):
        array =  np.concatenate([self.camera_target_loc[:,:,:2],
                                self.camera_query_radius[:,:,:1],
                                self.camera_dir[:,:,:2],
                                self.light_dir[:,:,:2] ],
                                axis=-1)

        array = array.reshape(-1).astype(np.float32)

        return array

    def generate_buffer(self, slice_index, num_levels, fixed_cam, fixed_light, fixed_patch):
        self.camera_target_loc, self.base_radius = self.generator.generate_camera_locations(fixed_patch)
        self.camera_query_radius = self.generator.generate_camera_patch_size(num_levels, self.base_radius, fixed_patch)
        self.camera_dir = self.generator.generate_dir(fixed_cam, slice_index, camera=True)
        self.light_dir = self.generator.generate_dir(fixed_light, slice_index, camera=False)


# Helper functions
def get_patch_size_linear(shape, num_resolution):
    patch_size = torch.rand(shape)*(num_resolution - 1)
    patch_size = torch.pow(2, patch_size)
    return patch_size

def get_patch_size_proportional(shape):
    y = torch.rand(shape)
    return 1/torch.sqrt(y+1e-6)

def get_patch_size_zero(shape):
    return torch.ones(shape)

def calc_origin(target_loc, dir):
    return target_loc - dir / np.absolute(dir[:, :, 2:]) * 10.0

def create_scrambled_stratas(n):
    stratas = torch.arange(n)
    rand_ids = torch.randperm(n)
    row_strata = stratas[rand_ids]
    rand_ids = torch.randperm(n)
    col_strata = stratas[rand_ids]

    rgrid, cgrid = torch.meshgrid(row_strata, col_strata, indexing='ij')

    return torch.dstack([rgrid, cgrid])

class GeneratorHelper:
    def __init__(self, res, num_slices) -> None:
        self.n = int(math.sqrt(num_slices))
        # assert(int(self.n + 0.5) ** 2 == num_slices) # num_slices needs to be a perfect square
        self.resolution = res

        self.camdir_stratas = create_scrambled_stratas(self.n)
        self.lightdir_stratas = create_scrambled_stratas(self.n)

    @staticmethod
    def calc_base_radius(scene_height):
        return 1 / scene_height / 2

    def generate_dir(self, fixed, slice_index, camera):
        height, width = self.resolution
        shape = [height, width, 1]

        row = int(slice_index / self.n)
        col = int(slice_index % self.n)

        if fixed:
            radius = torch.zeros(shape)
            theta = torch.zeros(shape)
        else:
            r1 = torch.rand(shape)
            r2 = torch.rand(shape)

            if camera:
                idx = self.camdir_stratas[row][col]
            else:
                idx = self.lightdir_stratas[row][col]

            r1 = (r1 + idx[0]) / self.n
            r2 = (r2 + idx[1]) / self.n
            theta = r1 * np.pi * 2
            radius = torch.sqrt(r2*.99)
            # Uncomment for constant slices of input directions
            # half_height = int(height / 2)
            # theta[:half_height, :, :] = theta[0, 0, 0]
            # theta[half_height:,  :,:] = theta[height - 1, width -1, 0]
            # radius[:half_height, :, :] = radius[0, 0, 0]
            # radius[half_height:,  :,:] = radius[height - 1, width -1, 0]

        dir_y = radius * torch.sin(theta)
        dir_x = radius * torch.cos(theta)

        dir_z = -torch.sqrt(1 - (dir_x*dir_x + dir_y*dir_y))

        dir = torch.cat([dir_x, dir_y, dir_z], -1)
        dir = dir.float()
        dir = dir.data.cpu().numpy()
        dir = np.ascontiguousarray(dir)

        return dir

    def camera_generate_origin(self, fixed_patch):
        height, width = self.resolution

        half_hor = 0.5/width
        half_ver = 0.5/height

        # 0 to 1
        loc_u = torch.linspace(half_hor, 1 - half_hor, width)
        loc_v = torch.linspace(half_ver, 1 - half_ver, height)

        loc_u, loc_v = torch.meshgrid(loc_u, 1 - loc_v, indexing='xy')
        loc = torch.stack([loc_u, loc_v], -1)

        loc = loc.float()
        loc = loc.data.cpu().numpy()

        # TODO: Figure out what's up with scene_resolution here
        sr_width, sr_height = self.resolution
        assert sr_width == sr_height

        base_radius = self.calc_base_radius(sr_height)

        return loc, base_radius

    def generate_camera_locations(self, fixed_patch):
        target_loc, base_radius = self.camera_generate_origin(fixed_patch)

        # origin = calc_origin(target_loc, dir)

        return target_loc, base_radius

    def generate_camera_patch_size(self, num_levels, base_radius, fixed_patch):
        height, width = self.resolution
        shape = [height, width, 1]

        masks = utils.tensor.get_masks([.3, .6, .1], shape)
        patch_size = masks[0] * get_patch_size_linear(shape, num_levels) + \
            masks[1] * get_patch_size_proportional(shape) + \
            masks[2] * get_patch_size_zero(shape)

        if fixed_patch:
            patch_size = torch.ones_like(patch_size)

        patch_size = patch_size * base_radius
        patch_size = patch_size.data.cpu().numpy()
        patch_size = np.ascontiguousarray(patch_size)

        return patch_size