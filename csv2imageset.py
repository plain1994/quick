#!/usr/bin/env python
# coding:utf-8
import os
from glob import glob
from dask import bag
import ast
from tqdm import tqdm

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw

NUM_CLASSES = 34
IMGS_PER_CLASS = 5000
IMG_HEIGHT, IMG_WIDTH = 64, 64

INPUT_DIR = "/media/workspace/yuchuan/quick/train_simplified/"
OUTPUT_DIR = "/media/workspace/yuchuan/quick/quickimgs/"

# class_files = os.listdir("../input/train_simplified/")
# classes = {x[:-4]:i for i, x in enumerate(class_files)}
# to_class = {i:x[:-4].replace(" ", "_") for i, x in enumerate(class_files)}

# faster conversion function
def draw_it(strokes):
    image = Image.new("P", (256, 256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i],
                             stroke[1][i],
                             stroke[0][i+1],
                             stroke[1][i+1]],
                            fill=0, width=5)
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    return np.array(image)


def csv2imgs(path):
    save_dir = os.path.join(OUTPUT_DIR, path.split(".")[0].replace(" ", "_"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_name = os.path.join(INPUT_DIR, path)
    train = pd.read_csv(csv_name, usecols=['drawing', 'recognized'])
    train = train[train.recognized == True].head(IMGS_PER_CLASS)
    image_bag = bag.from_sequence(train.drawing.values).map(draw_it)
    train_array = np.array(image_bag.compute())
    print(train_array.shape)
    for i in range(len(train_array)):
        img = Image.fromarray(train_array[i], mode="P")
        img.save(os.path.join(save_dir, str(i) + ".png"))
    print(csv_name)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    class_files = os.listdir(INPUT_DIR)
    num2class = {i: v[:-4].replace(" ", "_") for i, v in enumerate(class_files)}
    class2num = {v[:-4]: i for i, v in enumerate(class_files)}

    for i, p in enumerate(tqdm(class_files)):
        csv_name = p
        csv2imgs(csv_name)

    # csv2imgs(class_files[0])
        





if __name__ == "__main__":
    main()