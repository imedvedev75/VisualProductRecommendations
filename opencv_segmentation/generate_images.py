import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="input",
                    help="input image file")
parser.add_argument("--background_images", type=str, default="background_images",
                    help="list of background images")
parser.add_argument("--output_folder", type=str, default="output",
                    help="folder containing generated images")
parser.add_argument("--num_images", type=int, default=100,
                    help="number of image")
args = parser.parse_args()

input_file = args.input_file
background_images = args.background_images
output_folder = args.output_folder
num_images = args.num_images

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

for i in range(100):
    #if i % 12 != 0:
    #    continue
    command_to_run = "opencv_createsamples -img %s -bg %s -info %s/%s -maxxangle 1.1 -maxyangle 1.1 -maxzangle 0.5 -num %s -bgcolor 255" % (
    input_file, background_images, output_folder, i, num_images)
    os.system(command_to_run)

