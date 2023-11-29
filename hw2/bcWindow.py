from operator import truediv
from pickle import FALSE, TRUE
import tkinter as tk
from tkinter import Toplevel, Scale, Label, Button
import cv2
import numpy as np


class BcWindow(Toplevel):

    def __init__(self, master=None):
        Toplevel.__init__(self, master=master)

        # toplevel window
        self.title("bright & contract")
        self.config(bg="#404040")

        self.edited_image = self.master.edited_image.copy()

        # for toggle diffrent method, the adjection will not be overlay
        self.for_toggle_image = self.master.edited_image.copy()

        self.temp_image = None

        self.again_image = None  # for do the same method, the adjection will not be overlay
        self.be_lin = FALSE
        self.be_exp = FALSE
        self.be_ln = FALSE

        # put scale in
        self.a_label = Label(self, text="alpha", fg='white',
                             font=('arial bold', 16))
        self.a_label.config(bg="#404040")
        self.a_scale = Scale(self, from_=-3,
                             to_=3, length=250, orient="horizontal", resolution=0.1)
        self.b_label = Label(self, text="beta", fg='white',
                             font=('arial bold', 16))
        self.b_label.config(bg="#404040")
        self.b_scale = Scale(self, from_=-50,
                             to_=50, length=250, orient="horizontal", resolution=0.1)

        self.a_scale.set(1)
        self.b_scale.set(1)
        self.a_label.pack()
        self.a_scale.pack()
        self.b_label.pack()
        self.b_scale.pack()

        def show_image(img=None):
            self.master.image_display.display_image(img=img)

        # I do not want the every adjustion is overlayed, I want it is "readjustion", so I use if...else

        def linerly():  # in linerly way
            if self.be_lin == TRUE:
                self.edited_image = cv2.convertScaleAbs(
                    self.again_image, alpha=self.a_scale.get(), beta=self.b_scale.get())
            else:
                self.again_image = self.edited_image
                self.edited_image = cv2.convertScaleAbs(
                    self.for_toggle_image, alpha=self.a_scale.get(), beta=self.b_scale.get())

                self.be_lin = TRUE

            self.master.edited_image = self.edited_image.copy()
            show_image(self.edited_image)

        def exponentially():  # in exponentially way, I change the formula to exp(aX)+b

            alpha = float(np.exp(self.a_scale.get()))
            # beta = float(np.exp(self.b_scale.get()))

            if self.be_exp == TRUE:

                self.temp_image = cv2.multiply(
                    self.again_image, np.array([alpha]))
                self.edited_image = cv2.add(
                    self.temp_image, self.b_scale.get())
            else:
                self.again_image = self.edited_image

                self.temp_image = cv2.multiply(
                    self.for_toggle_image, np.array([alpha]))
                self.edited_image = cv2.add(
                    self.temp_image, self.b_scale.get())

                self.be_exp = TRUE

            self.master.edited_image = self.edited_image.copy()
            show_image(self.edited_image)

        def logarithmically():  # in logarithmically way

            alpha = float(np.log(self.a_scale.get()))
            beta = float(np.log(self.b_scale.get()))

            if self.be_ln == TRUE:

                self.temp_image = cv2.multiply(
                    self.again_image, np.array([alpha]))
                self.edited_image = cv2.add(
                    self.temp_image, beta)
            else:
                self.again_image = self.edited_image

                self.temp_image = cv2.multiply(
                    self.for_toggle_image, np.array([alpha]))
                self.edited_image = cv2.add(
                    self.temp_image, beta)

                self.be_ln = TRUE

            self.master.edited_image = self.edited_image.copy()
            show_image(self.edited_image)

        # linerly button
        self.linerly_button = Button(
            self, text="linerly", command=linerly)
        self.linerly_button.pack()

        # exponentially button
        self.exponentially_button = Button(
            self, text="exponentially", command=exponentially)
        self.exponentially_button.pack()

        # logarithmically button
        self.logarithmically_button = Button(
            self, text="logarithmically beta > 1", command=logarithmically)
        self.logarithmically_button.pack()
