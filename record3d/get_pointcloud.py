#!/usr/bin/env python

#run with python get_point_cloud.py --input_file=[your r3d file]

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(
    description="A simple script that takes an input file as an argument"
)

# Add the input file argument
parser.add_argument("--input_file", help="Path to the r3d input file")
parser.add_argument(
    "--output_file", default="r3dpointcloud.ply", help="Path to the outputed ply file"
)

# Parse the arguments
args = parser.parse_args()

assert (
    args.output_file[-4:] == ".ply"
), "The output file needs to be a ply file, which means the file name needs to end in .ply"

# Print the input file path
print(f"{args.input_file=}")
print(f"{args.output_file=}")

from data_utils import get_pointcloud, get_posed_rgbd_dataset

get_pointcloud(
    get_posed_rgbd_dataset(key="r3d", path=args.input_file), args.output_file
)

print("... created " + args.output_file + ".")
