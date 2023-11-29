from bisect import bisect_right
from ssl import CHANNEL_BINDING_TYPES
import tkinter as tk
from tkinter import Frame, filedialog
import cv2
from numpy import dtype
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


class Basetool(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master=master)

        # function
        def open_action():
            filename = filedialog.askopenfilename()
            image = cv2.imread(filename)

            if image is not None:
                self.master.filename = filename
                self.master.original_image = image.copy()
                self.master.edited_image = image.copy()
                self.master.image_display.display_image()
                self.master.is_image_select = True

        def save_action():
            save_image = self.master.edited_image

            file_type = self.master.filename.split('.')[-1]
            filename = filedialog.asksaveasfilename()
            filename_of_image = filename + "." + file_type

            cv2.imwrite(filename_of_image, save_image)

            self.master.filename = filename

        def revert_action():
            self.master.edited_image = self.master.original_image

            self.master.image_display.display_image(
                img=self.master.edited_image)

        def average_action():

            # the first img
            fd = open('pirate_a.raw', 'rb')
            rows = 512
            cols = 512
            f = np.fromfile(fd, dtype=np.uint8, count=rows*cols)
            img = f.reshape((rows, cols))  # notice row, column format

            mask = np.ones([3, 3], dtype=int)
            mask = mask / 9  # 3*3 1/9 array

            # Convolve the 3X3 mask over the image
            img_new = np.zeros([rows, cols])

            # to avoid number of array being -1, so range from 1
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0] + img[i, j] * \
                        mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2,
                                                                                 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]

                    img_new[i, j] = temp

            img_new = img_new.astype(np.uint8)
            image = Image.fromarray(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
            # from array to RGB(It does work in PIL,but doesn't in cv2. I don't know why)
            image.show()

            # the second img
            fd2 = open('pirate_b.raw', 'rb')
            f2 = np.fromfile(fd2, dtype=np.uint8, count=rows*cols)
            img2 = f2.reshape((rows, cols))  # notice row, column format

            # Convolve the 3X3 mask over the image
            img2_new = np.zeros([rows, cols])

            # to avoid number of array being -1, so range from 1
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    temp = img2[i-1, j-1]*mask[0, 0]+img2[i-1, j]*mask[0, 1]+img2[i-1, j + 1]*mask[0, 2]+img2[i, j-1]*mask[1, 0] + img2[i, j] * \
                        mask[1, 1]+img2[i, j + 1]*mask[1, 2]+img2[i + 1, j-1]*mask[2,
                                                                                   0]+img2[i + 1, j]*mask[2, 1]+img2[i + 1, j + 1]*mask[2, 2]

                    img2_new[i, j] = temp

            img2_new = img2_new.astype(np.uint8)
            image2 = Image.fromarray(cv2.cvtColor(img2_new, cv2.COLOR_BGR2RGB))
            image2.show()

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def median_action():
            # the first img
            fd = open('pirate_a.raw', 'rb')
            rows = 512
            cols = 512
            f = np.fromfile(fd, dtype=np.uint8, count=rows*cols)
            img = f.reshape((rows, cols))  # notice row, column format

            # Convolve the 3X3 mask over the image
            img_new = np.zeros([rows, cols])

            # Traverse the image. For every 3X3 area
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    temp = [img[i-1, j-1],
                            img[i-1, j],
                            img[i-1, j + 1],
                            img[i, j-1],
                            img[i, j],
                            img[i, j + 1],
                            img[i + 1, j-1],
                            img[i + 1, j],
                            img[i + 1, j + 1]]

                    temp = sorted(temp)
                    # find the median of the pixels and replace the center pixel by the median
                    img_new[i, j] = temp[4]

            img_new = img_new.astype(np.uint8)
            image = Image.fromarray(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
            # from array to RGB(It does work in PIL,but doesn't in cv2. I don't know why)
            image.show()
            self.master.best_image = img_new

            # the second img
            fd2 = open('pirate_b.raw', 'rb')
            f2 = np.fromfile(fd2, dtype=np.uint8, count=rows*cols)
            img2 = f2.reshape((rows, cols))  # notice row, column format

            # Convolve the 3X3 mask over the image
            img2_new = np.zeros([rows, cols])
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    temp = [img2[i-1, j-1],
                            img2[i-1, j],
                            img2[i-1, j + 1],
                            img2[i, j-1],
                            img2[i, j],
                            img2[i, j + 1],
                            img2[i + 1, j-1],
                            img2[i + 1, j],
                            img2[i + 1, j + 1]]

                    temp = sorted(temp)
                    img2_new[i, j] = temp[4]

            img2_new = img2_new.astype(np.uint8)

            image2 = Image.fromarray(cv2.cvtColor(img2_new, cv2.COLOR_BGR2RGB))
            image2.show()

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def lapla_action():

            # originl image
            fd = open('pirate_a.raw', 'rb')
            rows = 512
            cols = 512
            f = np.fromfile(fd, dtype=np.uint8, count=rows*cols)
            img = f.reshape((rows, cols))  # notice row, column format
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image.show()

            # laplacian image
            fd2 = self.master.best_image
            fd2 = cv2.Laplacian(fd2, cv2.CV_16S, ksize=3)
            fd2 = cv2.convertScaleAbs(fd2)
            fd2 = fd2.astype(np.uint8)
            image2 = Image.fromarray(cv2.cvtColor(fd2, cv2.COLOR_BGR2RGB))
            image2.show()

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # button
        self.open_button = tk.Button(
            self, text="Open", command=open_action, relief="solid", borderwidth=2, font=('arial bold', 15))

        self.save_button = tk.Button(
            self, text="Save", command=save_action, relief="solid", borderwidth=2, font=('arial bold', 15))

        self.revert_button = tk.Button(
            self, text="Revert", command=revert_action, relief="solid", borderwidth=2, font=('arial bold', 15))

        self.average_button = tk.Button(
            self, text="average mask", command=average_action, relief="solid", borderwidth=2, font=('arial bold', 15))

        self.median_button = tk.Button(
            self, text="median fliter", command=median_action, relief="solid", borderwidth=2, font=('arial bold', 15))

        self.lapla_button = tk.Button(
            self, text="Laplacian mask", command=lapla_action, relief="solid", borderwidth=2, font=('arial bold', 15))
        # grid

        self.open_button.grid(
            row=0, column=1, rowspan=2, padx=15, pady=10, sticky='sw')

        self.save_button.grid(
            row=0, column=2, rowspan=2, padx=15, pady=10, sticky='sw')

        self.revert_button.grid(
            row=0, column=3, rowspan=2, padx=15, pady=10, sticky='sw')

        self.average_button.grid(
            row=2, column=1, rowspan=2, padx=15, pady=10, sticky='sw')

        self.median_button.grid(
            row=2, column=2, rowspan=2, padx=15, pady=10, sticky='sw')

        self.lapla_button.grid(
            row=2, column=3, rowspan=2, padx=15, pady=10, sticky='sw')
