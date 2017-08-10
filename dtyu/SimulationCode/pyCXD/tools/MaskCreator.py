import numpy as np
from matplotlib.pyplot import * 
from matplotlib.path import Path
#test of interaction
import time
import h5py

'''Right now it's very rudimentary.
'''

class MaskCreator:
    '''A Mask creator creates figure and keeps a pointer to it
            A click and drag drags image only if shift is not pressed.
            A mouse wheel zooms in and out.
            Can specify a figure number to avoid conflict.
            outfile - the mask to write to
            winnum - figure number (for matplotlib)
            ratio - the ratio of the non-selected to selected image
                (default : .5) set to zero to not see non selected image
            zoom_scale - scaling for zoom
    '''
    def __init__(self,outfile="mask.hd5",data=None,blemish=None,inmask=None,winnum=1005,ratio=.5,zoom_scale=2.):
        #test case to be removed in the future
        if(data is None):
            self.img = np.random.random((1000,1000))
        else:
            self.img = data

        shiftPressed = False
        self.outfile = outfile
        self.ratio = ratio
        self.zoom_scale = zoom_scale
        self.dims = self.img.shape
        #damn coordinates are flipped, matplotlib... :-|
        self.ydim,self.xdim = self.dims
        x = np.arange(self.xdim)
        y = np.arange(self.ydim)
        self.X,self.Y = np.meshgrid(x,y)
        self.fig = figure(winnum)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_autoscale_on = False
        #self.ax.set_xlim([-.5, self.xdim+.5])
        #self.ax.set_ylim([-.5, self.ydim+.5])
        self.xs = np.array([])
        self.ys = np.array([])
        self.roi = None
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        #self.fig.canvas.mpl_connect('scroll_event', self.zoom)
        self.fig.canvas.mpl_connect('key_press_event', self.menu)
        self.closed = False
        self.masking = False 
        if(blemish is None):
            self.blemish = np.ones(self.img.shape)
        else:
            self.blemish = blemish
        if(inmask is None):
            self.mask = np.zeros(self.img.shape)
        else:
            self.mask = inmask
        self.im = self.ax.imshow(self.img*self.mask)
        self.mode = 'none'
        self.X = self.X.ravel()
        self.Y = self.Y.ravel()
        self.allpoints = np.transpose([np.transpose(self.X),np.transpose(self.Y)]) #create an [[x,y],...] pair array
        self.clim0 = None
        self.clim1 = None
        print("Welcome to mask creator. Mask by default includes no points.\n\
                Middle mouse button drag moves image\n\
                Middle mouse button wheel zooms image\n\
                Key Menu (cursor must be focused in window before key press):\n\
                i - select a region to include\n\
                x - select a region to exclude\n\
                a - select all pixels\n\
                n - select no pixels\n\
                r - reset and view mask (and rescale to view)\n\
                q - quit\n\
                ")

    def set_clim(self,cmin,cmax):
        self.clim0 = cmin
        self.clim1 = cmax
        self.im.set_clim(cmin,cmax)
        self.redraw()

    def menu(self,event):
        if(event.key == 'q'):
            print("Quitting, clearing figure")
            self.done == True
        elif(event.key == 'i'):
            print("Preparing to include a region\n\
                    Use shift +left click to make a new point\n\
                    and shift + right click to close polygon and terminate.\n\
                    Middle click drag moves the image and mouse wheel zooms.\n")
            self.masking = True
            self.mode = 'i'
            self.closed = False
            self.clearpoints()
            self.redraw()
            #self.ax.set_xlim([-.5, self.xdim+.5])
            #self.ax.set_ylim([-.5, self.ydim+.5])
        elif(event.key == 'x'):
            print("Preparing to exclude a region\n\
                    Use shift +left click to make a new point\n\
                    and shift + right click to close polygon and terminate.\n\
                    Middle click drag moves the image and mouse wheel zooms.\n")
            self.masking = True
            self.mode = 'x'
            self.closed = False
            self.clearpoints()
            self.redraw()
            #self.ax.set_xlim([-.5, self.xdim+.5])
            #self.ax.set_ylim([-.5, self.ydim+.5])
        elif(event.key == 'a'):
            print("Selecting all pixels\n")
            self.mask = self.mask*0 + 1
            self.redraw()
        elif(event.key == 'n'):
            print("Removing all pixels\n")
            self.mask = self.mask*0
            self.redraw()
        elif(event.key == 'r'):
            print("Rescaling to view")
            self.clearpoints()
            self.redraw()

    def zoom(self,event):
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location

        if event.button == 'down':
            # deal with zoom in
            scale_factor = self.zoom_scale
        elif event.button == 'up':
            # deal with zoom out
            scale_factor = 1 / self.zoom_scale
        else:
            # deal with something that should never happen
            scale_factor = 1

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        self.ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        self.ax.figure.canvas.draw()

    def onclick(self, event):
        ''' Event handling of the click event. If:
            Left mouse + shift clicked : add a new point to the ROI
            Left mouse clicked : This is an ROI drag event. Save current coordinates
                (not implemented yet)
            Mouse wheel clicked : this is an image drag event. Save current coordinates.
            Right button + shift clicked : Finish masking and close polygon
            '''
        self.x0,self.y0 = event.x,event.y
        self.xdata0,self.ydata0 = event.xdata,event.ydata
        if(event.button == 1 and event.key == 'shift' and self.closed == False and self.masking == True):
            self.xs = np.append(self.xs, self.xdata0)
            self.ys = np.append(self.ys, self.ydata0)
            self.drawseg()
        elif(event.button == 3 and event.key == 'shift' and self.closed == False and self.masking == True and len(self.xs) > 0):
            self.xs = np.append(self.xs,self.xs[0])
            self.ys = np.append(self.ys,self.ys[0])
            self.closed = True
            self.masking = False 
            self.findpoints()
            if(self.mode == 'i'):
                self.includepoints()
                print("Finished including region\n")
            elif(self.mode == 'x'):
                self.excludepoints()
                print("Finished excluding region\n")
            self.clearpoints()
            self.redraw()
        #print("key={}, button={}, x={}, y={}, xdata={}, ydata={}".format(
        #event.key, event.button, event.x, event.y, event.xdata, event.ydata))

    def findpoints(self):
        ''' Find the points in a polygon.'''
        self.polypath = Path(np.transpose([self.xs,self.ys]))
        self.mtmp = self.polypath.contains_points(self.allpoints)
        self.mtmp.shape = self.dims

    def includepoints(self):
        self.mask = np.bitwise_or(self.mask.astype(bool),self.mtmp).astype(int)
        self.mask.shape = self.dims

    def excludepoints(self):
        self.mask = np.bitwise_and(self.mask.astype(bool),np.invert(self.mtmp)).astype(int)
        self.mask.shape = self.dims

    def clearpoints(self):
        ''' Clear the points recorded. Also clear temp mask and polygon created.'''
        self.xs = []
        self.ys = []
#        self.mtmp = []
        self.polypath = []

    def redraw(self):
        self.ax.clear()
        self.im = self.ax.imshow(self.img*self.mask*self.blemish + self.img*self.blemish*self.ratio)
        if(len(self.xs)>0):
            self.ax.plot(self.xs,self.ys,'g-',linewidth=1.0)
        #self.ax.set_xlim([-.5, self.xdim+.5])
        #self.ax.set_ylim([-.5, self.ydim+.5])
        if(self.clim0 is not None):
            self.im.set_clim(self.clim0,self.clim1)
        self.fig.canvas.draw()
                
    def drawseg(self):
        if(len(self.xs)>1):
            self.ax.plot([self.xs[-2],self.xs[-1]],[self.ys[-2],self.ys[-1]],'g-',linewidth=1.0)
        self.fig.canvas.draw()
    
    def onrelease(self, event):
        if(event.button == 2 and event.key != 'shift'):
            xlims = self.ax.get_xlim()
            ylims = self.ax.get_ylim()
            xdata1,ydata1 = event.xdata,event.ydata
            if(xdata1 != None and ydata1 != None and self.xdata0 != None and self.ydata0 != None):
                dx,dy = xdata1-self.xdata0,ydata1-self.ydata0
                self.ax.set_xlim(xlims[0] - dx, xlims[1] - dx)
                self.ax.set_ylim(ylims[0] - dy, ylims[1] - dy)
                self.fig.canvas.draw()
                #shiftPressed = False
                #print("key={}, button={}, x={}, y={}, xdata={}, ydata={}".format(
                #event.key, event.button, event.x, event.y, event.xdata, event.ydata))

#ion()
#a = MaskCreator()
