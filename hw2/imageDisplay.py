import tkinter as tk
from tkinter import Frame, Canvas
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageDisplay(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master=master)

        self.canvas = Canvas(self, bg="gray", width=500, height=700)
        self.canvas.grid(row=0, column=1)

    def display_image(self, img=None):
        self.canvas.delete("all")  # clear all

        if img is None:
            image = self.master.edited_image.copy()
        else:
            image = img

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = img[:, :, ::-1]  # bgr to rgb
        height, width, channels = image.shape[:3]
        ratio = height/width

        new_height = height
        new_width = width

        if height > 700 or width > 500:
            if ratio < 1:
                new_width = 500
                new_height = int(new_height * ratio)
            else:
                new_height = 700
                new_width = int(new_width * (1/ratio))

        self.ratio = height/new_height
        self.now_displayimage = cv2.resize(image, (new_width, new_height))
        self.now_displayimage = ImageTk.PhotoImage(Image.fromarray(
            self.now_displayimage))  # display jpg & from numpy array to PIL image

        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(
            new_width/2, new_height/2, image=self.now_displayimage, anchor="center")
