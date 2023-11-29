from bisect import bisect_right
from itertools import filterfalse
from pickle import FALSE, TRUE
import tkinter as tk
from tkinter import Frame, ttk
import cv2
import numpy as np
from matplotlib import pyplot as plt


from bcWindow import BcWindow


class Tool(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master=master)

        self.be_rotated = FALSE
        self.be_blur = FALSE
        self.be_sharpen = FALSE

        # function

        def brightness_action():
            # brightness window
            self.master.brightWindow = BcWindow(master=self.master)
            self.master.brightWindow.grab_set()

            # brightWindow.mainloop()

        def resize_action():
            height, width = self.master.original_image.shape[:2]
            w = int(width*self.size_number.get())
            h = int(height*self.size_number.get())
            dim = (w, h)
            self.edited_image = cv2.resize(
                self.master.edited_image, dim, interpolation=cv2.INTER_LINEAR)

            self.master.edited_image = self.edited_image.copy()

            self.master.image_display.display_image(
                img=self.master.edited_image)

        def rotate_action():
            height, width = self.master.edited_image.shape[:2]

            # I want to judge the picture have been rotate before, avoid center point from moving every time when I rotate
            if self.be_rotated == TRUE:
                R = cv2.getRotationMatrix2D(
                    (height/2, width/2), self.rotate_number.get(), 1)
                self.now_image = cv2.warpAffine(
                    self.now_image, R, (height, width))

                self.master.edited_image = self.now_image.copy()
            else:
                R = cv2.getRotationMatrix2D(
                    (height/2, width/2), self.rotate_number.get(), 1)
                self.edited_image = cv2.warpAffine(
                    self.master.edited_image, R, (height, width))

                self.now_image = self.edited_image.copy()
                self.master.edited_image = self.edited_image.copy()

            self.master.image_display.display_image(
                img=self.master.edited_image)

            self.be_rotated = TRUE

        def gray_levelOO_action():
            self.edited_image = self.master.edited_image
            row, colume = self.edited_image.shape[:2]

            self.slicing_img = np.zeros((row, colume), dtype='uint8')
            min_range = int(self.min_input.get())
            max_range = int(self.max_input.get())

            for i in range(row):
                for j in range(colume):
                    if self.edited_image[i, j][0] > min_range and self.edited_image[i, j][0] < max_range:
                        self.slicing_img[i, j] = 255
                    else:
                        self.slicing_img[i, j] = self.edited_image[i, j][0]

            self.master.edited_image = self.slicing_img
            self.master.image_display.display_image(self.master.edited_image)

        def gray_levelBB_action():

            self.edited_image = self.master.edited_image
            row, colume = self.edited_image.shape[:2]

            self.slicing_img = np.zeros((row, colume), dtype='uint8')
            min_range = int(self.min_input.get())
            max_range = int(self.max_input.get())

            for i in range(row):
                for j in range(colume):
                    if self.edited_image[i, j][0] > min_range and self.edited_image[i, j][0] < max_range:
                        self.slicing_img[i, j] = 255
                    else:
                        self.slicing_img[i, j] = 0

            self.master.edited_image = self.slicing_img
            self.master.image_display.display_image(self.master.edited_image)

        def show_histogram_action():
            plt.hist(self.master.edited_image.ravel(), 256, [0, 256])
            plt.show()

        def change_histogram_action():
            self.edited_image = self.master.edited_image
            self.edited_image = cv2.cvtColor(
                self.edited_image, cv2.COLOR_BGR2GRAY)
            self.edited_image = cv2.equalizeHist(
                self.edited_image)
            self.master.edited_image = self.edited_image
            self.master.image_display.display_image(self.master.edited_image)

        def bit_plane_action():
            self.edited_image = self.master.edited_image
            lst = []
            row, colume = self.edited_image.shape[:2]
            # change pixel value to binary using np.binary_repr() and store in the list
            for i in range(row):
                for j in range(colume):
                    lst.append(np.binary_repr(
                        self.edited_image[i][j][0], width=8))  # width is 8 bit
            bit_plane_num = int(self.bit_plane_number.get()
                                )  # which bit plane image
            bit_plane_num = bit_plane_num - 1
            pixel_value = 7 - bit_plane_num
            bit = int(2 ** bit_plane_num)

            self.edited_image = (np.array([int(i[pixel_value]) for i in lst], dtype='uint8')
                                 * bit).reshape(self.edited_image.shape[0], self.edited_image.shape[1])

            self.edited_image = cv2.normalize(self.edited_image, np.zeros(
                self.edited_image.shape), 0, 255, cv2.NORM_MINMAX)

            self.master.edited_image = self.edited_image.copy()
            self.master.image_display.display_image(self.master.edited_image)

        def blur_action():

            self.edited_image = self.master.edited_image

            if self.be_blur == TRUE:

                self.edited_image = cv2.blur(
                    self.now_image, (self.blur_number.get(), self.blur_number.get()))

                self.master.edited_image = self.edited_image.copy()
            else:
                self.now_image = self.edited_image.copy()

                self.edited_image = cv2.blur(
                    self.edited_image, (self.blur_number.get(), self.blur_number.get()))

                self.master.edited_image = self.edited_image.copy()

                self.be_blur = TRUE

            self.master.image_display.display_image(
                img=self.master.edited_image)

        def sharpen_action():
            self.edited_image = self.master.edited_image
            alpha = int(self.sharpen_number.get())
            beta = 1 - alpha

            if self.be_sharpen == TRUE:
                self.blur_image = cv2.GaussianBlur(self.now_image, (0, 0), 3)

                self.edited_image = cv2.addWeighted(
                    self.now_image, alpha, self.blur_image, beta, 0)

                self.master.edited_image = self.edited_image.copy()

            else:
                self.now_image = self.edited_image.copy()

                self.blur_image = cv2.GaussianBlur(
                    self.edited_image, (0, 0), 3)

                self.edited_image = cv2.addWeighted(
                    self.edited_image, alpha, self.blur_image, beta, 0)

                self.master.edited_image = self.edited_image.copy()
                self.be_sharpen = TRUE

            self.master.image_display.display_image(
                img=self.master.edited_image)

        # button

        self.brightness_button = tk.Button(
            self, text="Bright & Contrast", command=brightness_action, relief="solid", borderwidth=2, font=('arial bold', 15))

        self.size_button = tk.Button(
            self, text="Resize Ok", command=resize_action, relief="solid", borderwidth=2, font=('arial bold', 15))
        self.size_number = tk.Scale(self, from_=0,
                                    to_=3, length=250, label="zoom in or shrink", orient="horizontal", resolution=0.2)
        self.size_number.set(1)

        self.rotate_button = tk.Button(
            self, text="Rotate Ok", command=rotate_action, relief="solid", borderwidth=2, font=('arial bold', 15))
        self.rotate_number = tk.Scale(self, from_=0,
                                      to_=180, length=250, label="rotate degrees", orient="horizontal", resolution=1)
        self.rotate_number.set(1)

        self.graylevelOO_button = tk.Button(
            self, text="Gray Level Slicing (original value)", command=gray_levelOO_action, relief="solid", borderwidth=2, font=('arial bold', 10))
        self.graylevelBB_button = tk.Button(
            self, text="Gray Level Slicing (black color)", command=gray_levelBB_action, relief="solid", borderwidth=2, font=('arial bold', 10))

        self.min_input = tk.Entry(self)
        self.min_input.config(font=('arial bold', 15))
        self.min_input.insert(0, 'Enter min range')

        self.max_input = tk.Entry(self)
        self.max_input.config(font=('arial bold', 15))
        self.max_input.insert(0, 'Enter max range')

        self.histogram_button = tk.Button(
            self, text="Show histogram", command=show_histogram_action, relief="solid", borderwidth=2, font=('arial bold', 15))

        self.change_histogram_button = tk.Button(
            self, text="auto-level", command=change_histogram_action, relief="solid", borderwidth=2, font=('arial bold', 15))

        self.bit_plane_button = tk.Button(
            self, text="bit-plane this", command=bit_plane_action, relief="solid", borderwidth=2, font=('arial bold', 15))
        self.bit_plane_number = tk.Scale(self, from_=1,
                                         to_=8, length=250, label="which one bit-plane image", orient="horizontal", resolution=1)
        self.rotate_number.set(0)

        self.blur_button = tk.Button(
            self, text="blur this", command=blur_action, relief="solid", borderwidth=2, font=('arial bold', 15))
        self.blur_number = tk.Scale(self, from_=1,
                                    to_=25, length=250, label="how blur you want?", orient="horizontal", resolution=1)
        self.blur_number.set(12)

        self.sharpen_button = tk.Button(
            self, text="sharpen this", command=sharpen_action, relief="solid", borderwidth=2, font=('arial bold', 15))
        self.sharpen_number = tk.Scale(self, from_=1,
                                       to_=10, length=250, label="how sharp you want?", orient="horizontal", resolution=0.5)
        self.sharpen_number.set(5)

        # grid

        self.brightness_button.grid(
            row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nw')

        self.size_button.grid(
            row=1, column=0, columnspan=2, padx=10, pady=10, sticky='nw')

        self.size_number.grid(
            row=2, column=0, sticky='nw')

        self.rotate_button.grid(
            row=3, column=0, columnspan=2, padx=10, pady=10, sticky='nw')

        self.rotate_number.grid(
            row=4, column=0, sticky='nw')

        self.graylevelOO_button.grid(
            row=1, column=1, columnspan=2, padx=15, pady=10, ipady=5, sticky='nw')

        self.graylevelBB_button.grid(
            row=2, column=1, columnspan=2, padx=15, pady=10, ipady=5, sticky='nw')

        self.min_input.grid(
            row=3, column=1, columnspan=1, padx=15, pady=10, ipadx=1, ipady=3, sticky='nw')
        self.max_input.grid(
            row=4, column=1, columnspan=1, padx=15, pady=10, ipadx=1, ipady=3, sticky='nw')

        self.histogram_button.grid(
            row=6, column=0, columnspan=2, padx=10, pady=10, sticky='nw')
        self.change_histogram_button.grid(
            row=6, column=1, columnspan=2, padx=10, pady=10, sticky='nw')

        self.bit_plane_button.grid(
            row=7, column=0, columnspan=2, padx=10, pady=10, sticky='nw')
        self.bit_plane_number.grid(
            row=8, column=0, sticky='nw')

        self.blur_button.grid(
            row=9, column=0, columnspan=2, padx=10, pady=10, sticky='nw'
        )
        self.blur_number.grid(
            row=10, column=0, sticky='nw')

        self.sharpen_button.grid(
            row=9, column=1, columnspan=2, padx=10, pady=10, sticky='nw'
        )
        self.sharpen_number.grid(
            row=10, column=1,  padx=10, sticky='nw')
