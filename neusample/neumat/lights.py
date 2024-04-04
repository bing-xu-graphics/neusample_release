from unittest import result
import torch as th
import torch.linalg as tla

class PointLight:
    def __init__(self, position, intensity) -> None:
        self.intensity = intensity
        self.position = position
        self.device = f"cuda:0"

    def sample(self, shading_locations):
        if len(shading_locations.shape) == 2:
            pos_z = th.zeros(shading_locations.shape[0], 1).to(self.device)
            shading_locations = th.cat([shading_locations, pos_z], dim=-1)

        # positions = self.position.repeat(shading_locations.shape)
        positions = th.zeros_like(shading_locations)
        positions[:,:] = self.position
        result_dirs = positions - shading_locations
        dist = tla.norm(result_dirs, dim=-1)
        dir_x = result_dirs[:,0].div(dist)
        dir_y = result_dirs[:,1].div(dist)
        dir_z = result_dirs[:,2].div(dist)
        return th.dstack([dir_x, dir_y, dir_z]), self.intensity / (dist * dist)