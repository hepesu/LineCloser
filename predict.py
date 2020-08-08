import os

# Try running on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import cv2
from keras.models import load_model

R = 2 ** 4
MODEL_NAME = './model1.h5'

model = load_model(MODEL_NAME)
model.summary()

for root, dirs, files in os.walk('./input', topdown=False):
    for name in files:
        print(os.path.join(root, name))

        im = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)

        im_predict = cv2.resize(im, (im.shape[1] // R * R, im.shape[0] // R * R))
        im_predict = np.reshape(im_predict, (1, im_predict.shape[0], im_predict.shape[1], 1))
        im_predict = im_predict.astype(np.float32) / 255.

        result = model.predict(im_predict)

        im_res = cv2.resize(result[0] * 255., (im.shape[1], im.shape[0]))

        cv2.imwrite(os.path.join('./output', name), im_res)
