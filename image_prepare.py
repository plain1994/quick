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
IMG_HEIGHT, IMG_WIDTH = 32, 32


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
    return np.array(image)/255.


def main():
    print("hello")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    train_grand = []
    class_paths = glob('/media/workspace/yuchuan/quick/train_simplified/*.csv')

    for i, c in enumerate(tqdm(class_paths[0:NUM_CLASSES])):
        # for i,c in enumerate(tqdm(class_paths[0:2])):
        train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=IMGS_PER_CLASS * 5 // 4)
        train = train[train.recognized == True].head(IMGS_PER_CLASS)
        imagebag = bag.from_sequence(train.drawing.values).map(draw_it)
        trainarray = np.array(imagebag.compute())
        trainarray = np.reshape(trainarray, (IMGS_PER_CLASS, -1))
        labelarray = np.full((train.shape[0], 1), i)
        trainarray = np.concatenate((labelarray, trainarray), axis=1)
        train_grand.append(trainarray)

    print(train.shape)
    # 5000, 2
    print(trainarray.shape)
    # 5000, 1025

    train_grand = np.array([train_grand.pop() for i in np.arange(NUM_CLASSES)])
    print(train_grand.shape)
    # 34, 5000, 1025
    train_grand = train_grand.reshape((-1, (IMG_HEIGHT * IMG_WIDTH + 1)))
    print(train_grand.shape)
    # 170000, 1025

    del trainarray
    del train

    valfrac = 0.1

    cutpt = int(valfrac * train_grand.shape[0])

    np.random.shuffle(train_grand)

    y_train, X_train = train_grand[cutpt:, 0], train_grand[cutpt:, 1:]
    print(y_train.shape)
    # 153000
    print(X_train.shape)
    # 153000, 1024
    y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:]
    print(y_val.shape)
    # 17000
    print(X_val.shape)
    # 17000, 1024

    del train_grand



if __name__ == "__main__":
    main()