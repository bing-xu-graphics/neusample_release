import argparse, os, sys
from ntpath import join

import numpy as np
import torch as th

# Insert root dir in path
p = os.path.abspath('.')
sys.path.insert(1, p)
from neumat import interfaces, lights, datasets

class NPZExporter:
    def __init__(self) -> None:
        pass

    def export(self, model: interfaces.NeuMIPv1Module, outdir=None):
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, model.conf.name + ".npz")
        print("### Exporting model " + model.conf.name + " to file: " + out_path)

        dtype = np.float32
        saved_model = {}
        for n, p in model.named_parameters():
            if n.endswith("texture"):
                tex = p.detach().cpu()
                tex = tex.permute(1, 2, 0)
                tex_0 = tex[:, :, :4]
                tex_1 = tex[:, :, 4:]
                saved_model[n + "_half1"] = tex_0.numpy().astype(dtype)
                saved_model[n + "_half2"] = tex_1.numpy().astype(dtype)
            else:
                saved_model[n] = p.detach().cpu().numpy().astype(dtype)

        np.savez(out_path, **saved_model)

        return out_path

    def test_load(self, out_path):

        mynpfile = np.load(out_path)
        print("### Model layers ###")
        print(mynpfile.files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="models")

    args = parser.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.exists(ckpt):
        raise ValueError("Checkpoint doesn't exist")

    model = interfaces.NeuMIPv1Module.load_from_checkpoint(ckpt, strict=False)

    exporter = NPZExporter()
    exported_file = exporter.export(model, args.output)
    exporter.test_load(exported_file)



if __name__ == "__main__":
    main()