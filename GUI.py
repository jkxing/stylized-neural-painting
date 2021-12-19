import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
import matplotlib.pyplot as plt
import numpy as np
import cv2

class GUI(tk.Frame):
    def __init__(self, parent = None):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.img_path = ''
        self.save_path = ''

        self.frame0 = tk.Frame(self, bd = 10)
        self.frame0.pack()
        self.path_label = tk.Label(self.frame0, text = '')
        self.path_label.pack(side='left')
        self.browseButton = tk.Button(self.frame0, text = 'Browse', command = self.openfile)
        self.browseButton.pack(side = 'left')
        self.slider_var = tk.IntVar()
        self.slider = tk.Scale(self, from_=1, to=20, orient= 'horizontal', variable = self.slider_var, command = self.slider_changed)
        self.slider.pack(pady = 10)

        self.goButton = tk.Button(self, text = 'Paint', command = self.go, width = 20)
        self.goButton.pack(pady = 10)

        self.addButton = tk.Button(self, text = 'Add Area', command = self.add_area, width = 20)
        self.addButton.pack(pady = 10)

        self.saveButton = tk.Button(self, text = 'Save as...', command = self.savefile, width = 20)
        self.saveButton.pack(pady = 10)


        self.mark_val = 1
        self.oval_size = 1
    
    def paint(self, event):
        python_green = "#476042"       

        x1, y1 = ( event.x - self.oval_size ), ( event.y - self.oval_size )
        x2, y2 = ( event.x + self.oval_size ), ( event.y + self.oval_size )
        for x in range(x1, x2+1) :
            for y in range(y1, y2 + 1):
                self.image_mask[y][x][0] = self.mark_val
                self.image_mask[y][x][1] = self.mark_val
                self.image_mask[y][x][2] = self.mark_val

        self.canvas.create_oval( x1, y1, x2, y2, fill = python_green )
    
    def add_area(self):
        self.mark_val += 1
    
    def slider_changed(self, event):
        self.oval_size = self.slider_var.get()
        # print(self.slider_var.get())

    def go(self):
        if (len(self.img_path) == 0):
            mb.showinfo('No image selected', 'Please browse an image to be resized')
            return
        # img = plt.imread(self.img_path)
        img = ImageTk.PhotoImage(Image.open(self.img_path))

        offspring = tk.Toplevel()
        offspring.title(self.img_path.split('/')[-1])
        offspring.geometry('%sx%s' % (img.width()+10, img.height()+10))
        self.image_mask = np.zeros((img.height(), img.width(), 3))
        self.canvas = tk.Canvas(offspring, width=img.width(), height=img.height(),
                   borderwidth=0, highlightthickness=0)
        self.canvas.pack(expand=True)
        self.canvas.img = img  # Keep reference in case this code is put into a function.
        self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        self.canvas.bind( "<B1-Motion>", self.paint )
        offspring.mainloop()

    def openfile(self):
        self.img_path = fd.askopenfilename()
        self.path_label.config(text = self.img_path) 

    def savefile(self):
        self.save_path = fd.asksaveasfilename()
        if len(self.save_path) == 0 :
            mb.showinfo('Give destination', 'Please give a destination path')
            return

        cv2.imwrite(self.save_path, self.image_mask)
        with open(self.save_path[:-4]+'.npy', 'wb') as f:
            np.save(f, np.array(self.image_mask)) 

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('%sx%s' % (400, 300))
        
    gui = GUI(root)
    gui.pack()
    root.mainloop()