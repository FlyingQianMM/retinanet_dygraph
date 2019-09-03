#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import os, sys
import os.path as osp
import glob
import shutil


def convert_model(input_dir):
    for name in glob.glob(input_dir + '/*'):
        if name.split('/')[-1] == 'optimizers':
            shutil.rmtree(name)
            continue

        if os.path.isfile(name):
            continue

        if os.path.isdir(name):
            for file in glob.glob(name + '/*'):
                file_name = file.split('.')[-1]
                shutil.move(file, input_dir + '/' + file_name)
            shutil.rmtree(name)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', help='directory of the model to be converted')

    args = parser.parse_args()
    print(args.model_dir)
    convert_model(args.model_dir)
