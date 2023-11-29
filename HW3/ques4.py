from skimage.color import label2rgb
import numpy as np
import cv2
from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, Frame, LEFT, RIGHT
from math import sqrt, cos, acos, degrees, radians, pi
from scipy import ndimage as nd

window = tk.Tk()
window.title('Color Image')
button_frame = Frame(window)


def RGB():
    img = imread('Lenna_512_color.tif')
    x, y, z = np.shape(img)
    # new matrix let the rgb of original image turn into 0
    red = np.zeros((x, y, z), dtype=int)
    green = np.zeros((x, y, z), dtype=int)
    blue = np.zeros((x, y, z), dtype=int)
    for i in range(0, x):
        for j in range(0, y):
            # only remain R from original img, gb are 0
            red[i][j][0] = img[i][j][0]
            green[i][j][1] = img[i][j][1]  # only remain G
            blue[i][j][2] = img[i][j][2]  # only remain B

    plt.subplot(131).set_title("Red")
    plt.imshow(red)
    plt.subplot(132).set_title("Green")
    plt.imshow(green)
    plt.subplot(133).set_title("Blue")
    plt.imshow(blue)
    plt.show()


def HSI_to_bgr(h, s, i):
    h = degrees(h)
    if 0 <= h <= 120:
        b = i * (1 - s)
        r = i * (1 + (s * cos(radians(h)) / cos(radians(60) - radians(h))))
        g = i * 3 - (r + b)
    elif 120 < h <= 240:
        h -= 120
        r = i * (1 - s)
        g = i * (1 + (s * cos(radians(h)) / cos(radians(60) - radians(h))))
        b = 3 * i - (r + g)
    elif 0 < h <= 360:
        h -= 240
        g = i * (1 - s)
        b = i * (1 + (s * cos(radians(h)) / cos(radians(60) - radians(h))))
        r = i * 3 - (g + b)
    return [b, g, r]


def rgb_to_hue(b, g, r):
    if (b == g == r):
        return 0

    angle = 0.5 * ((r - g) + (r - b)) / \
        sqrt(((r - g) ** 2) + (r - b) * (g - b))
    if b <= g:
        return acos(angle)
    else:
        return 2 * pi - acos(angle)


def rgb_to_intensity(b, g, r):
    val = (b + g + r) / 3.
    if val == 0:
        return 0
    else:
        return val


def rgb_to_saturity(b, g, r):
    if r + g + b != 0:
        return 1. - 3. * np.min([r, g, b]) / (r + g + b)
    else:
        return 0


def HSI():
    img = imread('Lenna_512_color.tif')
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    plt.subplot(131).set_title("Hue")
    plt.imshow(hsv_image[:, :, 0], cmap="gray", vmin=0, vmax=255)
    plt.subplot(132).set_title("Saturation")
    plt.imshow(hsv_image[:, :, 1], cmap="gray", vmin=0, vmax=255)
    plt.subplot(133).set_title("Intensity")
    plt.imshow(hsv_image[:, :, 2], cmap="gray", vmin=0, vmax=255)
    plt.show()


def Complement():
    img = imread('Lenna_512_color.tif')
    # There are two ways to realize it

    # first way
    # h, w, c = img.shape[:3]
    # size = (h, w, c)
    # neg_image = np.zeros(size, np.uint8)

    # for i in range(0, h):
    #     for j in range(0, w):
    #         for y in range(0, c):
    #             # nagetive the color every pixel
    #             neg_image[i, j, y] = 255 - img[i, j, y]

    # second way
    img = 255 - img

    plt.imshow(img)
    plt.show()


def Smooth():
    img = cv2.imread('Lenna_512_color.tif')
    # rgb smooth
    smooth_rbg_img = cv2.blur(img, (5, 5))
    smooth_rbg_img = cv2.cvtColor(smooth_rbg_img, cv2.COLOR_BGR2RGB)

    # hsi smooth
    height, width = img.shape[0], img.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
    I = np.zeros((height, width))
    S = np.zeros((height, width))
    H = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            b = img[i][j][0] / 255.
            g = img[i][j][1] / 255.
            r = img[i][j][2] / 255.
            H[i][j] = rgb_to_hue(b, g, r)
            S[i][j] = rgb_to_saturity(b, g, r)
            I[i][j] = rgb_to_intensity(b, g, r)

    I1 = cv2.blur(I, (5, 5))  # smooth intensity

    # hsi to bgr
    for i in range(height):
        for j in range(width):
            bgr_tuple = HSI_to_bgr(H[i][j], S[i][j], I1[i][j])

            new_image[i][j][0] = np.clip(round(bgr_tuple[0] * 255.), 0, 255)
            new_image[i][j][1] = np.clip(round(bgr_tuple[1] * 255.), 0, 255)
            new_image[i][j][2] = np.clip(round(bgr_tuple[2] * 255.), 0, 255)

    smooth_hsi_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    # laplacian rbg
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lapla_rbg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]])

    lapla_rbg_img = cv2.filter2D(lapla_rbg_img, -1, kernel_sharpening)
    lapla_rbg_img = cv2.addWeighted(lapla_rbg_img, 0.3, img, 1, 0.0)

    # laplacian hsi
    I2 = cv2.filter2D(I, -1, kernel_sharpening)
    # hsi to bgr
    new_image1 = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            bgr_tuple = HSI_to_bgr(H[i][j], S[i][j], I2[i][j])

            new_image1[i][j][0] = np.clip(round(bgr_tuple[0] * 255.), 0, 255)
            new_image1[i][j][1] = np.clip(round(bgr_tuple[1] * 255.), 0, 255)
            new_image1[i][j][2] = np.clip(round(bgr_tuple[2] * 255.), 0, 255)

    new_image1 = cv2.cvtColor(new_image1, cv2.COLOR_BGR2RGB)
    lapla_hsi_img = cv2.addWeighted(new_image1, 0.3, img, 1, 0.0)

    smooth_diff = 255 - cv2.absdiff(smooth_rbg_img, smooth_hsi_img)
    lapla_diff = 255 - cv2.absdiff(lapla_rbg_img, lapla_hsi_img)

    plt.subplot(231).set_title("Smooth in RGB")
    plt.axis('off')
    plt.imshow(smooth_rbg_img)
    plt.subplot(232).set_title("Smooth in HSI")
    plt.axis('off')
    plt.imshow(smooth_hsi_img)
    plt.subplot(233).set_title("Difference of two Smooth")
    plt.axis('off')
    plt.imshow(smooth_diff)
    plt.subplot(234).set_title("Laplacian in RGB")
    plt.axis('off')
    plt.imshow(lapla_rbg_img)
    plt.subplot(235).set_title("Laplacian in HSI")
    plt.axis('off')
    plt.imshow(lapla_rbg_img)
    plt.subplot(236).set_title("Difference of two Laplacian")
    plt.axis('off')
    plt.imshow(lapla_diff)
    plt.tight_layout()
    plt.show()


def Feather():
    img = imread('Lenna_512_color.tif')
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (110, 100, 100), (160, 255, 255))
    closed_mask = nd.binary_closing(mask, np.ones((7, 7)))
    image_overly = label2rgb(closed_mask, image=img)

    plt.subplot(221).set_title("1: Turn into HSV")
    plt.axis('off')
    plt.imshow(hsv)
    plt.subplot(222).set_title("2: Find proper mask")
    plt.axis('off')
    plt.imshow(mask)
    plt.subplot(223).set_title("3: Closed the mask")
    plt.axis('off')
    plt.imshow(closed_mask)
    plt.subplot(224).set_title("4: Result")
    plt.axis('off')
    plt.imshow(image_overly)

    plt.show()


RGB_button = tk.Button(
    button_frame, text="RGB", command=RGB, relief="solid", borderwidth=2, font=('arial bold', 15))
RGB_button.grid(
    row=0, column=1, rowspan=2, padx=15, pady=10, sticky='sw')

HSI_button = tk.Button(
    button_frame, text="HSI", command=HSI, relief="solid", borderwidth=2, font=('arial bold', 15))
HSI_button.grid(
    row=2, column=1, rowspan=2, padx=15, pady=10, sticky='sw')

Complement_button = tk.Button(
    button_frame, text="Color Complement", command=Complement, relief="solid", borderwidth=2, font=('arial bold', 15))
Complement_button.grid(
    row=4, column=1, rowspan=2, padx=15, pady=10, sticky='sw')

smooth_button = tk.Button(
    button_frame, text="Smooth & Sharpen", command=Smooth, relief="solid", borderwidth=2, font=('arial bold', 15))
smooth_button.grid(
    row=6, column=1, rowspan=2, padx=15, pady=10, sticky='sw')

feather_button = tk.Button(
    button_frame, text="Feather", command=Feather, relief="solid", borderwidth=2, font=('arial bold', 15))
feather_button.grid(
    row=8, column=1, rowspan=2, padx=15, pady=10, sticky='sw')

# original picture
original = Image.open("Lenna_512_color.tif")
original = ImageTk.PhotoImage(original)
ori_img = tk.Label(window, image=original)
ori_img.image = original

# window out
button_frame.pack(side=LEFT)
button_frame.config(bg="#404040")
ori_img.pack(side=RIGHT)
window.config(bg="#404040")
window.mainloop()
