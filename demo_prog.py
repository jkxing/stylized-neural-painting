import argparse
import torch
import torch.optim as optim
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
import matplotlib.pyplot as plt
import numpy as np
from painter import *
from GUI import GUI
# settings
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
parser.add_argument('--img_path', type=str, default='./test_images/apple.jpg', metavar='str',
                    help='path to test image (default: ./test_images/apple.jpg)')
parser.add_argument('--img_seg_path', type=str, default='', metavar='str')
parser.add_argument('--renderer', type=str, default='oilpaintbrush', metavar='str',
                    help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle (default oilpaintbrush)')
parser.add_argument('--canvas_color', type=str, default='black', metavar='str',
                    help='canvas_color: [black, white] (default black)')
parser.add_argument('--canvas_size', type=int, default=512, metavar='str',
                    help='size of the canvas for stroke rendering')
parser.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                    help='keep input aspect ratio when saving outputs')
parser.add_argument('--max_m_strokes', type=int, default=500, metavar='str',
                    help='max number of strokes (default 500)')
parser.add_argument('--max_divide', type=int, default=5, metavar='N',
                    help='divide an image up-to max_divide x max_divide patches (default 5)')
parser.add_argument('--beta_L1', type=float, default=1.0,
                    help='weight for L1 loss (default: 1.0)')
parser.add_argument('--with_ot_loss', action='store_true', default=False,
                    help='imporve the convergence by using optimal transportation loss')
parser.add_argument('--beta_ot', type=float, default=0.1,
                    help='weight for optimal transportation loss (default: 0.1)')
parser.add_argument('--net_G', type=str, default='zou-fusion-net-light', metavar='str',
                    help='net_G: plain-dcgan, plain-unet, huang-net, zou-fusion-net, '
                         'or zou-fusion-net-light (default: zou-fusion-net-light)')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'./checkpoints_G_oilpaintbrush_light', metavar='str',
                    help='dir to load neu-renderer (default: ./checkpoints_G_oilpaintbrush_light)')
parser.add_argument('--lr', type=float, default=0.002,
                    help='learning rate for stroke searching (default: 0.005)')
parser.add_argument('--output_dir', type=str, default=r'./output', metavar='str',
                    help='dir to save painting results (default: ./output)')
parser.add_argument('--disable_preview', action='store_true', default=False,
                    help='disable cv2.imshow, for running remotely without x-display')
args = parser.parse_args()

class GUI(tk.Frame):
    def __init__(self, parent = None):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.img_path = args.img_path
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
            
        args.img_seg_path = self.save_path[:-4]+'.npy'

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def optimize_x(pt):

    pt._load_checkpoint()
    pt.net_G.eval()

    print('begin drawing...')

    PARAMS = np.zeros([1, 0, pt.rderr.d], np.float32)
    final_img = np.zeros((args.canvas_size,args.canvas_size,3),dtype = np.uint8)
    print(pt.region_levels)
    #pt.region_levels=2
    
    video_writer = cv2.VideoWriter(
        pt.img_path[:-4]+"_animated.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 40,
        final_img.shape[:2])
    for level in range(0,pt.region_levels):
        if pt.rderr.canvas_color == 'white':
            CANVAS_tmp = torch.ones([1, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)
        else:
            CANVAS_tmp = torch.zeros([1, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)
        
        pt.mask = (pt.img_seg==level)
        pos = np.where(pt.mask==1)
        d = np.max(pos[0])
        r = np.max(pos[1])
        u = np.min(pos[0])
        l = np.min(pos[1])
        if level!=0:
            CANVAS_tmp = final_img[u:d+1,l:r+1,:].astype(np.float32)/255.0
            CANVAS_tmp = utils.img2patches(CANVAS_tmp, 1, pt.net_G.out_size).to(device)
        area = (d-u+1)*(r-l+1)
        print("d,r,u,l",d,r,u,l)
        print("area",area)
        pt.max_divide = math.ceil(math.sqrt(area)/pt.net_G.out_size)
        siz = pt.max_divide*pt.net_G.out_size
        lis = list(range(1,pt.max_divide+1))
        t = len(lis)
        while t<4:
            lis.append(pt.max_divide)
            t = t+1
        lis.append(pt.max_divide)
        print(lis)
        pt.img_ = np.copy(pt.img__[u:d+1,l:r+1,:])
        #pt.img_ = pt.img__ * pt.mask
        if level==0:
            if pt.rderr.canvas_color == 'white':
                pt.img_[~pt.mask] = 255
            else:
                pt.img_[~pt.mask] = 0
        #pt.img_ = pt.img__ #origin
        PARAM = np.zeros([1, 0, pt.rderr.d], np.float32)
        cnt = 0
        for pt.m_grid in lis[:-1]:
            cnt += 1
            pt.img_batch = utils.img2patches(pt.img_, pt.m_grid, pt.net_G.out_size).to(device)
            pt.G_final_pred_canvas = CANVAS_tmp

            pt.initialize_params()
            pt.x_ctt.requires_grad = True
            pt.x_color.requires_grad = True
            pt.x_alpha.requires_grad = True
            utils.set_requires_grad(pt.net_G, False)

            pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr, centered=True)

            pt.step_id = 0
            #pt.m_strokes_per_block = 1
            for pt.anchor_id in range(0, pt.m_strokes_per_block):
                pt.stroke_sampler(pt.anchor_id)
                iters_per_stroke = int(500 / pt.m_strokes_per_block)
                #iters_per_stroke = 1
                for i in range(iters_per_stroke):
                    pt.G_pred_canvas = CANVAS_tmp
                    # update x
                    pt.optimizer_x.zero_grad()

                    pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                    pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                    pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                    pt._forward_pass()
                    pt._drawing_step_states()
                    pt._backward_x()

                    pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                    pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                    pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                    pt.optimizer_x.step()
                    pt.step_id += 1

            v = pt._normalize_strokes(pt.x)
            v = pt._shuffle_strokes_and_reshape(v)
            PARAM = np.concatenate([PARAM, v], axis=1)
            CANVAS_tmp = pt._render(PARAM, save_jpgs=False, save_video=False)
            CANVAS_tmp = utils.img2patches(CANVAS_tmp, lis[cnt], pt.net_G.out_size).to(device)
        PARAMS = np.concatenate([PARAMS, PARAM], axis=1)
        img = pt._render(PARAM, save_jpgs=False, save_video=False, suffix = str(level))*255
        img = img.astype(np.uint8)
        pt.rderr.canvas = cv2.resize(final_img[u:d+1,l:r+1,:3].astype(np.float32)/255.0, (pt.rderr.CANVAS_WIDTH,pt.rderr.CANVAS_WIDTH))
        cv2.imshow("before render",pt.rderr.canvas)
        v = PARAM[0,:,:]
        for i in range(v.shape[0]):  # for each stroke
            pt.rderr.stroke_params = v[i, :]
            if pt.rderr.check_stroke():
                pt.rderr.draw_stroke()
            this_frame = pt.rderr.canvas
            this_frame = cv2.resize(this_frame,(r-l+1,d-u+1))*255
            this_frame = this_frame.astype(np.uint8)
            final_img[u:d+1,l:r+1,:] = this_frame
            #if level==0:
            #    final_img[(pt.img_seg!=level)]=255
            timg = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            video_writer.write(timg.astype(np.uint8))
        #final_img[u:d+1,l:r+1,:] = final_img[u:d+1,l:r+1,:]*(1-mask[u:d+1,l:r+1,:]) + cv2.resize(img,(r-l+1,d-u+1))*mask[u:d+1,l:r+1,:]
    pt._save_stroke_params(PARAMS)
    #final_rendered_image = pt._render(PARAMS, save_jpgs=True, save_video=True)
    pt.rderr.create_empty_canvas() 
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    cv2.imshow("final_img",final_img)
    cv2.imwrite("qiu_out.jpg",final_img)
    cv2.waitKey(0)



if __name__ == '__main__':
    if args.img_seg_path=="":
        root = tk.Tk()
        root.geometry('%sx%s' % (400, 300))
        gui = GUI(root)
        gui.pack()
        root.mainloop()
    pt = ProgressivePainter(args=args)
    optimize_x(pt)

