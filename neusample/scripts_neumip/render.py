import argparse, os, sys
from turtle import width

import numpy as np
import torch as th
from torch.utils.data import DataLoader

# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)
from neumat import interfaces, lights, datasets
from utils import exr, ops, la2

class Renderer:
    def __init__(self, resolution, neumat, dataset=None, device="cuda:0"):
        self.width = resolution[0]
        self.height = resolution[1]
        self.device = device
        self.model = neumat
        self.dataset = dataset

    def render_btf_queries(self, frames, outdir):
        if self.dataset is None or not os.path.exists(self.dataset):
            raise ValueError("Valid input dataset required to visualize btf queries")

        neumipdata = datasets.NeuMIPv1Dataset(self.dataset)

        # Visualize inputs
        neumipdata.dump_h5(outdir, frames)

        dataloader = DataLoader(neumipdata, batch_size=1, num_workers=0, shuffle=False)

        curr_frame = 0
        for batch_idx, batch in enumerate(dataloader):
            camera_dir, light_dir, color, uv, cam_qr = batch

            camera_dir = camera_dir.reshape(-1, camera_dir.shape[-1])
            print(camera_dir.shape)
            light_dir = light_dir.reshape(-1, light_dir.shape[-1])
            uv = uv.reshape(-1, uv.shape[-1])
            color = color.reshape(-1, color.shape[-1])

            rgb_prediction = self.model(camera_dir, light_dir, uv)

            rgb_prediction = rgb_prediction.reshape(self.width, self.height, -1).cpu()
            outfile = os.path.join(outdir, "pred_" + str(curr_frame) + ".exr")
            exr.write16(np.array(rgb_prediction, dtype=np.float32), outfile)

            curr_frame += 1
            break
            if (curr_frame == frames):
                break



    def render_with_point_lights(self, frames, outdir):
        half_texel_size = 0.5 / self.width

        pos_coord = th.linspace(half_texel_size, 1 - half_texel_size, self.width)
        posu, posv = th.meshgrid(pos_coord, 1 - pos_coord, indexing='xy')
        uv = th.dstack((posu, posv))
        uv = uv.reshape(-1, 2).to(self.device)
        print(uv)

        normals = th.zeros(self.width * self.height, 3).to(self.device)
        normals[:,2] = 1 # z-up
        # camera_dir = th.zeros_like(uv).to(self.device)
        camera_dir = th.ones_like(uv).to(self.device)

        for i in range(frames):

            pos = th.zeros(1, 1, 3)
            pos[:, :, 0] = -3 + i * 6.0/(frames-1)
            pos[:, :, 1] = 0
            pos[:, :, 2] = 3
            point_light = lights.PointLight(pos, 10)
            world_space_shading_pts = uv * 2 - 1
            light_dir, light_sample = point_light.sample(world_space_shading_pts)
            light_dir.squeeze_()

            btf = self.model(camera_dir, light_dir[:, :2], uv)

            rgb_prediction = la2.dot(normals, light_dir).squeeze()
            rgb_prediction = rgb_prediction * light_sample
            r = btf[:, 0] * rgb_prediction
            g = btf[:, 1] * rgb_prediction
            b = btf[:, 2] * rgb_prediction
            pred = th.dstack([r, g, b]).cpu()
            pred = pred.reshape(self.width, self.height, -1).cpu()
            outfile = os.path.join(outdir, "pred_" + str(i) + ".exr")
            exr.write16(np.array(pred, dtype=np.float32), outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["pointlight", "btf"], default="btf")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="/home/bingxu/projects/NeuMIP/res/datasets/shell.hdf5")
    parser.add_argument("--output", default=os.path.join("renders/fabric_moving_light"), help="dir to render outputs")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--resolution", default=[512, 512]) #type=list[int], 
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    device = "cpu"
    if th.cuda.is_available():
        device = f"cuda:0"
    else:
        raise ValueError("CPU not supported")

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.exists(ckpt):
        raise ValueError("Checkpoint doesn't exist")

    model = interfaces.NeuMIPv1Module.load_from_checkpoint(ckpt, strict=False)

    model.model.initial_sigma = 1
    model.model.iterations_to_sigma_1 = 1

    conf = model.conf

    if args.verbose:
        print("Model parameters:")
        for n, p in model.named_parameters():
            print(f"  {n}, {p.mean().item():.4f} {p.std().item():.4f}")

    model.to(device)
    model.eval()
    model.freeze()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.mode == "pointlight":
        renderer = Renderer(args.resolution, model, dataset=None, device=device)
        renderer.render_with_point_lights(args.frames, args.output)
    elif args.mode == "btf":
        renderer = Renderer(args.resolution, model, dataset=args.dataset, device=device)
        renderer.render_btf_queries(args.frames, args.output)
    else:
        raise ValueError("Unsupported render mode")
    rgb_tex = output_neural_texture(model)
    output_dir = "../pytorch_render/"
    exr.write32(np.array(rgb_tex, dtype=np.float32), os.path.join(output_dir, "gt_rgb_texture_%d_2.exr"%(1000)))

def output_neural_texture(module):
    rgb_tex = module.model.rgb_texture.texture[3:6, :, :].cpu()
    rgb_tex = rgb_tex.permute(1, 2, 0)
    return rgb_tex


if __name__ == "__main__":
    main()