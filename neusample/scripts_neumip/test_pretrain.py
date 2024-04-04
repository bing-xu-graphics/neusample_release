import argparse, os, sys
from turtle import width
import torch
import numpy as np
import torch as th
from torch.utils.data import DataLoader

# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)
from neumat import interfaces, lights, datasets
from utils import exr, ops, la2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["pointlight", "btf"], default="btf")
    # parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="../tortoise_shell.ckpt")
    # parser.add_argument("--checkpoint", type=str, default="/home/bingxu/projects/neumip_adobe/stylized_wool.ckpt")
    
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

    ### load the trained model
    checkpoint = torch.load(ckpt)
    print(checkpoint.keys())
    """
    dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 
    'state_dict', 'loops', 'callbacks', 'optimizer_states', 
    'lr_schedulers', 'hparams_name', 'hyper_parameters'])"""
    state_dict = checkpoint["state_dict"]
    """
    model.offset_texture.texture     torch.Size([8, 512, 512])
    model.offset_network.layers.0.weight     torch.Size([32, 10])
    model.offset_network.layers.0.bias       torch.Size([32])
    model.offset_network.layers.2.weight     torch.Size([32, 32])
    model.offset_network.layers.2.bias       torch.Size([32])
    model.offset_network.layers.4.weight     torch.Size([32, 32])
    model.offset_network.layers.4.bias       torch.Size([32])
    model.offset_network.layers.6.weight     torch.Size([1, 32])
    model.offset_network.layers.6.bias       torch.Size([1])
    model.rgb_texture.texture        torch.Size([8, 512, 512])
    model.rgb_network.layers.0.weight        torch.Size([32, 12])
    model.rgb_network.layers.0.bias          torch.Size([32])
    model.rgb_network.layers.2.weight        torch.Size([32, 32])
    model.rgb_network.layers.2.bias          torch.Size([32])
    model.rgb_network.layers.4.weight        torch.Size([32, 32])
    model.rgb_network.layers.4.bias          torch.Size([32])
    model.rgb_network.layers.6.weight        torch.Size([3, 32])
    model.rgb_network.layers.6.bias          torch.Size([3])
    """
    
    print("Model's state_dict:")
    for param_tensor in state_dict:
        print(param_tensor, "\t", state_dict[param_tensor].size())

    # model = interfaces.NeuMIPv1Module.load_from_checkpoint(ckpt, strict=False)

    # model.model.initial_sigma = 1
    # model.model.iterations_to_sigma_1 = 1

    # conf = model.conf

    # if args.verbose:
    #     print("Model parameters:")
    #     for n, p in model.named_parameters():
    #         print(f"  {n}, {p.mean().item():.4f} {p.std().item():.4f}")

    # model.to(device)
    # model.eval()
    # model.freeze()

    if not os.path.exists(args.output):
        os.makedirs(args.output)


def output_neural_texture(module):
    rgb_tex = module.model.rgb_texture.texture[3:6, :, :].cpu()
    rgb_tex = rgb_tex.permute(1, 2, 0)


if __name__ == "__main__":
    main()