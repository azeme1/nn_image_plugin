from __future__ import print_function
from gimpfu import *
import os
import subprocess
from time import sleep
import tempfile
import shutil


version_dict = {0: "Sketch2Cat_v0_small",
                1: "Sketch2Cat_v1",
                2: "Sketch2Cat_v2"}

def plugin_main(image, layer, version_index):
    debug = False
    plugin_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plugin_working_path = tempfile.mkdtemp()
    lib_path = os.path.join(plugin_root_path, "", "lib")
    bin_path = os.path.join(plugin_root_path, "", "bin")
    version = version_dict[version_index]
    version = version + "/" + version
    graph_path = os.path.join(plugin_root_path, "", "models/pix2pix/zaidalyafeai_zaidalyafeai_github_io/" + version + ".param")
    weight_path = os.path.join(plugin_root_path, "","models/pix2pix/zaidalyafeai_zaidalyafeai_github_io/" + version + ".bin")
    in_tensor_name, out_tensor_name = "data", "activation_9ltanh_0"


    executable = os.path.join(bin_path, '', 'ncnn_inference_runner.exe')
    os.environ['PATH'] = os.environ['PATH'] + ";" + lib_path

    file_path_src = os.path.join(plugin_working_path, "", "_src.png")
    file_path_dst = os.path.join(plugin_working_path, "", "_dst.png")
    pdb.file_jpeg_save(image, layer, file_path_src, file_path_src, 0.9, 0, 0, 0, "", 0, 0, 0, 0)

    args = (executable, graph_path, weight_path, in_tensor_name, out_tensor_name,
            file_path_src, file_path_dst, "256", "256", "0", "2", "0", "2")

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=debug)
    popen.wait()
    output = popen.communicate()
    print(output)

    mask = pdb.gimp_file_load_layer(image, file_path_dst)
    pdb.gimp_image_add_layer(image, mask, 0)
    shutil.rmtree(plugin_working_path, ignore_errors=True)


register(
    "python_fu_nnplugin_sketch2cat",
    N_("Convert Sketch to the cat picture"),
    "Run Neural Network inference to cat pictures from sketches (see https://github.com/azeme1/nn_image_plugin and https://github.com/zaidalyafeai/zaidalyafeai.github.io).",
    "azemel",
    "azemel",
    "2020",
    "<Image>/Filters/NNPlugin/Pix2Pix/Sketch2Cat",
    "RGB*, GRAY*",
    [
        (PF_RADIO, "version_index", "Select model version ", 0,
            (
                 ("Tiny Sketch2Cat model", 0),
                 ("Sketch2Cat Version 1", 1),
                 ("Sketch2Cat Version 2", 2)
            )
         )
    ],
    [],
    plugin_main)

main()