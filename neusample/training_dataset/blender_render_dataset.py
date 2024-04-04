import argparse
import bpy
import numpy as np
import sys, os
import h5py
import utils

from pathlib import Path
from tqdm import tqdm

TMP_OUTFILE = "temp_rgb.exr"

def extract_output_data(outexr):
    rgb = utils.exr.read(outexr)
    h, w, d = rgb.shape
    rgb = rgb.reshape(w * h, d)
    return rgb.tolist()

class BlenderRender:
    def __init__(self, out_path="", num_images=0) -> None:
        self.out_path = out_path
        self.num_images = num_images

        # Create h5 output file
        outdir = Path(self.out_path).parent
        if not outdir.exists():
            Path.mkdir(outdir)
            print("Created output directory: ", outdir)

        self.h5file = h5py.File(self.out_path, 'w')

    def write_input_dataset(self, input_dataset_path, resolution, base_radius):
        # Read inputs from query file
        input_data = np.fromfile(input_dataset_path, dtype=np.float32)
        inputs_query_size = 7
        num_queries = int(len(input_data) / inputs_query_size)
        input_data = input_data.reshape(num_queries, inputs_query_size)

        # Write out to h5 dataset
        def create_dataset_helper(name, channels, input_data):
            self.h5file.create_dataset(name, (num_queries, channels), dtype=np.float16, data=input_data)

        create_dataset_helper('ground_camera_target_loc', 2, input_data[:, 0:2])
        create_dataset_helper('ground_camera_query_radius', 1, input_data[:, 2:3])
        create_dataset_helper('ground_camera_dir', 2, input_data[:, 3:5])
        create_dataset_helper('ground_light_dir', 2, input_data[:, 5:7])

        self.h5file.attrs["base_radius"] = float(base_radius)
        self.h5file.attrs["resolution"] = resolution

    def render_single(self):
        # TODO: Figure out alternative, this is killing persistent data!!!
        bpy.context.scene.update_render_engine()
        bpy.ops.render.render(write_still=True)

    def render_dataset(self, query_file):
        scene = bpy.context.scene
        scene.world.cycles.query_filename = query_file

        render = bpy.context.scene.render
        render.image_settings.file_format = 'OPEN_EXR'
        render.image_settings.color_depth = '16'
        render.filepath = TMP_OUTFILE

        out_data = []
        for i in tqdm(range(self.num_images)):
            bpy.context.scene.world.cycles.query_index = i

            self.render_single()

            # Load rendered image
            curr_out_data = extract_output_data(TMP_OUTFILE)
            os.rename(TMP_OUTFILE, "renders/" + os.path.splitext(TMP_OUTFILE)[0] + str(i) + ".exr")
            out_data += curr_out_data

        out_data_array = np.array(out_data)
        # Write out output data to dataset
        self.h5file.create_dataset('ground_color', out_data_array.shape, dtype=np.float16, data=out_data_array)

def render(blend_file, query_file, out_path, num_images, resolution, base_radius):
    assert(Path(blend_file).exists())

    bpy.ops.wm.open_mainfile(filepath=blend_file)
    br = BlenderRender(out_path, num_images)

    br.write_input_dataset(query_file, resolution, base_radius)

    br.render_dataset(query_file)
    bpy.ops.wm.quit_blender()

if __name__ == "__main__":
    argv = sys.argv
    if ('--' in argv):
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend_file", type=str)
    parser.add_argument("--query_file", type=str)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--out_path", type=str)

    args = parser.parse_known_args(argv)[0]
    render(args.blend_file, args.query_file, args.out_path, args.num_images)