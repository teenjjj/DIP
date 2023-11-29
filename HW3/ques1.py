import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.io import imread


img = imread('BarTest.tif')

# 7*7 mean
ksize = 7
kernel = np.ones((ksize, ksize))
kernel /= kernel.sum()
img77 = convolve2d(img, kernel)

# 3*3 mean
ksize = 3
kernel = np.ones((ksize, ksize))
kernel /= kernel.sum()
img33 = convolve2d(img, kernel)

# 7*7 median
me_img77 = cv2.medianBlur(img, 7)

# 3*3 median
me_img33 = cv2.medianBlur(img, 3)


plt.subplot(221).set_title('7*7 arithmetic mean')
plt.gray()
plt.imshow(img77)

plt.subplot(222).set_title('3*3 arithmetic mean')
plt.imshow(img33)

plt.subplot(223).set_title('7*7 median')
plt.imshow(me_img77)

plt.subplot(224).set_title('3*3 median')
plt.imshow(me_img33)


plt.tight_layout()
plt.show()
