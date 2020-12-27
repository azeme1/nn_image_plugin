from __future__ import print_function
from gimpfu import *
import os
import subprocess
import tempfile
import shutil


def plugin_main(image, layer):
    debug = False
    plugin_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plugin_working_path = tempfile.mkdtemp()
    lib_path = os.path.join(plugin_root_path, "", "lib")
    bin_path = os.path.join(plugin_root_path, "", "bin")
    graph_path = os.path.join(plugin_root_path, "", "models/segmentation/ItchyHiker_Hair_Segmentation_Keras/CelebA_PrismaNet_256_hair_seg_model_opt_001.param")
    weight_path = os.path.join(plugin_root_path, "", "models/segmentation/ItchyHiker_Hair_Segmentation_Keras/CelebA_PrismaNet_256_hair_seg_model_opt_001.bin")

    model_url = "https://github.com/azeme1/nn_image_plugin/raw/dev/models/pix2pix/zaidalyafeai_zaidalyafeai_github_io/Sketch2Cat_v0_small/Sketch2Cat_v0_small.zip"
    in_tensor_name, out_tensor_name = "data", "softmax_convlsigmoid_0"
    get_model(model_url, [graph_path, weight_path])

    executable = os.path.join(bin_path, '', 'ncnn_inference_runner.exe')
    os.environ['PATH'] = os.environ['PATH'] + ";" + lib_path

    file_path_src = os.path.join(plugin_working_path, "", "_src.png")
    file_path_dst = os.path.join(plugin_working_path, "", "_dst.png")
    pdb.file_jpeg_save(image, layer, file_path_src, file_path_src, 0.9, 0, 0, 0, "", 0, 0, 0, 0)

    args = (executable, graph_path, weight_path, "data", in_tensor_name, out_tensor_name,
            file_path_src, file_path_dst, "256", "256", "0", "4", "1", "3")

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=debug)
    popen.wait()
    output = popen.communicate()
    print(output)

    mask = pdb.gimp_file_load_layer(image, file_path_dst)
    pdb.gimp_image_add_layer(image, mask, 0)
    shutil.rmtree(plugin_working_path, ignore_errors=True)


register(
    "python_fu_nnplugin",
    N_("Add a layer with hair segmentation mask (see https://github.com/azeme1/nn_image_plugin)"),
    "Run Neural Network inference to produce the hair segmentation mask (see https://github.com/azeme1/nn_image_plugin).",
    "azemel",
    "azemel",
    "2020",
    "<Image>/Filters/NNPlugin/Segmentation/Hair",
    "RGB*, GRAY*",
    [],
    [],
    plugin_main)

main()