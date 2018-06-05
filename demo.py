from mnist import MNIST
import random
import os
import cv2
import numpy as np
import math
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

mndata = MNIST('samples')
images, labels = mndata.load_training()
#images, labels = mndata.load_testing()
index = random.randrange(0, len(images))  # choose an index ;-)

SHAPE_NORM_TARGET_LENGTH = 64
PAD_PIXELS = 8
STEP_BLOCK = 8
LENGTH = SHAPE_NORM_TARGET_LENGTH + 2 * PAD_PIXELS
TOTAL_SIZE = LENGTH * LENGTH
NUMBER_OF_BLOCK_PER_ROW = (LENGTH - STEP_BLOCK) / STEP_BLOCK
RAW_FEATURE_BLOCK_SIZE = 32
FEATURE_BLOCK_SIZE = 16
FEATURE_SIZE = NUMBER_OF_BLOCK_PER_ROW * NUMBER_OF_BLOCK_PER_ROW * FEATURE_BLOCK_SIZE
FEATURE_REDUCE_SIZE = 160
W = (64,64)
thumbnail_size = (LENGTH,LENGTH)

ROOT_PATH = path = os.getcwd()
TRAIN_DATA_DIR = os.path.join(ROOT_PATH, "nhap/a.jpg")
#out_image = np.zeros((32,32),dtype=np.float32)
img = Image.open(TRAIN_DATA_DIR)
img_gray = img.convert('L')
#img = cv2.imread(TRAIN_DATA_DIR)
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img_gray.resize((64,64))

#print type(img)

background = Image.new('RGBA', (80,80), "black")
background.paste(img, (8,8))

plt.imshow(background)
plt.show()
background = np.array(background)
result = background[:, :, 0]
#print result.shape

ret,thresh2 = cv2.threshold(result,110,255,cv2.THRESH_BINARY_INV)
#out_image = cv2.normalize(thresh2.astype(np.float32), out_image, alpha=0, beta=255,
#
#                                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#plt.imshow(thresh2)
#plt.show()
#print(mndata.display(images[index]))

#print type(thresh2)
#print thresh2.size
#robert
roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

roberts_cross_h = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )

np.asarray( thresh2, dtype="int32" )
vertical = ndimage.convolve( thresh2, roberts_cross_v )
horizontal = ndimage.convolve( thresh2, roberts_cross_h )

Gradient_strength = np.sqrt( np.square(horizontal) + np.square(vertical))
Gradient_direction = np.arctan2(roberts_cross_h, roberts_cross_v)

#Gradient_strength = Gradient_strength1.flatten()

#9x9 subareas
#for r in np.arange(0,thresh2.shape[0] -PAD_PIXELS, PAD_PIXELS):
#    for c in range(0,thresh2.shape[1]- PAD_PIXELS, PAD_PIXELS):
#        window = thresh2[r:r+thumbnail_size[0],c:c+thumbnail_size[1]]
#        print window.shape


def FindHistogram():
    for i in range(0, LENGTH / STEP_BLOCK -1, 1):
        for j in range(0, LENGTH - STEP_BLOCK, STEP_BLOCK):
            ExtractBlockFeatures(i, j)

def ExtractBlockFeatures(i , j):
    i = i * STEP_BLOCK
    aS1 = FindGradientSumS1(i, j)
    aS2 = FindGradientSumS2(i, j)
    aS3 = FindGradientSumS3(i, j)
    aS4 = FindGradientSumS4(i, j)
    strength = 4 * aS1 + 3 * aS2 + 2 * aS3 + aS4

    stopRow = i + 16
    stopCol = j + 16
    twoPi = 2 * math.pi
    kArray = []
    iRow = i
    while iRow < stopRow:
        jCol = j
        while jCol < stopCol:
            k = (Gradient_direction[iRow * LENGTH + jCol] + math.pi) * 32 / twoPi
            kInt = math.floor(k)
            kRem = (k - kInt)
            if kInt > 31:
                kInt -= 32
            kArray[kInt] += strength * (1 - kRem)
            kInt += 1
            if kInt > 31:
                kInt -= 32
            kArray[kInt] += strength * kRem

def FindGradientSumS1(rowIndex, columnIndex):
    sum = 0
    row = (rowIndex + 6) * LENGTH + 6 + columnIndex
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3]

    return sum

def FindGradientSumS2(index, columnIndex):
    sum = 0
    row = (index + 4) * LENGTH + 4 + columnIndex
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7]

    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + [row + 6] + Gradient_strength[row + 7]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 6] + Gradient_strength[row + 7]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 6] + Gradient_strength[row + 7]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 6] + Gradient_strength[row + 7]

    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7]

    return sum

def FindGradientSumS3(index, columnIndex):
    sum = 0
    row = (index + 2) * LENGTH + 2 + columnIndex
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7] + Gradient_strength[row + 8] + Gradient_strength[row + 9] + Gradient_strength[row + 10] + \
           Gradient_strength[row + 11]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7] + Gradient_strength[row + 8] + Gradient_strength[row + 9] + Gradient_strength[row + 10] + \
           Gradient_strength[row + 11]

    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 10] + Gradient_strength[row + 11]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 10] + Gradient_strength[row + 11]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 10] + Gradient_strength[row + 11]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 10] + Gradient_strength[row + 11]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 10] + Gradient_strength[row + 11]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 10] + Gradient_strength[row + 11]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 10] + Gradient_strength[row + 11]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 10] + Gradient_strength[row + 11]

    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7] + Gradient_strength[row + 8] + Gradient_strength[row + 9] + Gradient_strength[row + 10] + \
           Gradient_strength[row + 11]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7] + Gradient_strength[row + 8] + Gradient_strength[row + 9] + Gradient_strength[row + 10] + \
           Gradient_strength[row + 11]


    return sum

def FindGradientSumS4(index, columnIndex):
    sum = 0
    row = index * LENGTH + columnIndex
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7] + Gradient_strength[row + 8] + Gradient_strength[row + 9] + Gradient_strength[row + 10] + \
           Gradient_strength[row + 11] + Gradient_strength[row + 12] + Gradient_strength[row + 13] + Gradient_strength[
               row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7] + Gradient_strength[row + 8] + Gradient_strength[row + 9] + Gradient_strength[row + 10] + \
           Gradient_strength[row + 11] + Gradient_strength[row + 12] + Gradient_strength[row + 13] + Gradient_strength[
               row + 14] + Gradient_strength[row + 15]

    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 14] + Gradient_strength[row + 15]

    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7] + Gradient_strength[row + 8] + Gradient_strength[row + 9] + Gradient_strength[row + 10] + \
           Gradient_strength[row + 11] + Gradient_strength[row + 12] + Gradient_strength[row + 13] + Gradient_strength[
               row + 14] + Gradient_strength[row + 15]
    row += LENGTH
    sum += Gradient_strength[row] + Gradient_strength[row + 1] + Gradient_strength[row + 2] + Gradient_strength[row + 3] + \
           Gradient_strength[row + 4] + Gradient_strength[row + 5] + Gradient_strength[row + 6] + Gradient_strength[
               row + 7] + Gradient_strength[row + 8] + Gradient_strength[row + 9] + Gradient_strength[row + 10] + \
           Gradient_strength[row + 11] + Gradient_strength[row + 12] + Gradient_strength[row + 13] + Gradient_strength[
               row + 14] + Gradient_strength[row + 15]


    return sum

model = FindHistogram()
print model