import tkinter as tk
from tkinter import ttk, LEFT, RIGHT

from tool import Tool
from basetool import Basetool
from imageDisplay import ImageDisplay


class Main(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.filename = ""
        self.orignal_image = None
        self.edited_image = None
        self.is_image_select = False
        self.best_image = None

        # welcome frontend
        self.title("Jean's image processing tool")
        self.config(bg="#404040")  # background
        tk.Label(text='My Photo Tool!', bg='#404040',
                 fg='white', font=('arial bold', 20)).pack()
        line1 = ttk.Separator(master=self, orient=tk.HORIZONTAL)
        line1.pack(fill=tk.X, padx=20, pady=5)

        # base tool pack
        self.basetool = Basetool(master=self)
        self.basetool.pack(pady=10)
        self.basetool.config(bg="#404040")

        # tool pack(
        self.tool = Tool(master=self)
        self.tool.pack(pady=10, side=LEFT)
        self.tool.config(bg="#404040")

        # canvas
        self.image_display = ImageDisplay(master=self)
        self.image_display.pack(padx=20, pady=10, expand=1, side=RIGHT)
