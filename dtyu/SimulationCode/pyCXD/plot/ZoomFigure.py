#create a new figure, zoom figure
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

class ZoomFigure:
    '''An object to extend matplotlib to scroll wheel zooming.
    #ref - taken from tacaswell
    http://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel
    Sets a zcid to figures to keep track of cid's from zoom figure'''
    def __init__(self,winnum=None,clear=0,logxy=[0,0],figsize=None):
        if(winnum is None):
            self.fig = plt.gcf()
        else:
            self.fig = plt.figure(winnum,figsize=figsize)

        if(clear == 1):
            self.clear()
        self.ax = self.fig.add_subplot(111)
        self.logxy(logxy)
        self.ax.set_aspect('auto',adjustable='box')
        self.ax.autoscale(tight=True)
        self.winnum = winnum
        if(not hasattr(self.fig,'zcids')):
            zcids = [self.fig.canvas.mpl_connect('button_press_event', self.onclick)]
            zcids.append(self.fig.canvas.mpl_connect('scroll_event', self.zoom))
            zcids.append(self.fig.canvas.mpl_connect('button_release_event', self.onrelease))
            zcids.append(self.fig.canvas.mpl_connect('motion_notify_event', self.onrelease))
            self.fig.zcids = zcids
        #get a stack overflow when i drag for too long so will ignore for now
        self.zoom_scale = 1.5

    def clear(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('auto',adjustable='box')
        self.ax.autoscale(tight=True)
        plt.sca(self.ax)

        if(hasattr(self.fig,'zcids')):
            self.fig.canvas.mpl_disconnect(self.fig.zcids[0])
            self.fig.canvas.mpl_disconnect(self.fig.zcids[1])
            self.fig.canvas.mpl_disconnect(self.fig.zcids[2])
            self.fig.canvas.mpl_disconnect(self.fig.zcids[3])

        zcids = [self.fig.canvas.mpl_connect('button_press_event', self.onclick)]
        zcids.append(self.fig.canvas.mpl_connect('scroll_event', self.zoom))
        zcids.append(self.fig.canvas.mpl_connect('button_release_event', self.onrelease))
        zcids.append(self.fig.canvas.mpl_connect('motion_notify_event', self.onrelease))
        self.fig.zcids = zcids


    def logxy(self,lxy):
        '''Change axes to log or non-log scale.'''
        lx = lxy[0]
        ly = lxy[1]
        if(lx == 1):
            if(ly == 1):
                self.ax.loglog()
            else:
                self.ax.semilogx()
        else:
            if(ly == 1):
                self.ax.semilogy()

    def zoom(self,event):
        '''This function zooms the image upon scrolling the mouse wheel.
        Scrolling it in the plot zooms the plot. Scrolling above or below the
        plot scrolls the x axis. Scrolling to the left or the right of the plot
        scrolls the y axis. Where it is ambiguous nothing happens. 
        NOTE: If expanding figure to subplots, you will need to add an extra
        check to make sure you are not in any other plot. It is not clear how to
        go about this.
        Since we also want this to work in loglog plot, we work in axes
        coordinates and use the proper scaling transform to convert to data
        limits.
        '''
        #convert pixels to axes
        tranP2A     = self.ax.transAxes.inverted().transform
        #convert axes to data limits
        tranA2D     = self.ax.transLimits.inverted().transform
        #convert the scale (for log plots)
        tranSclA2D  = self.ax.transScale.inverted().transform

        if event.button == 'down':
            # deal with zoom in
            scale_factor = self.zoom_scale
        elif event.button == 'up':
            # deal with zoom out
            scale_factor = 1 / self.zoom_scale
        else:
            # deal with something that should never happen
            scale_factor = 1

        #get my axes position to know where I am with respect to them
        xa,ya = tranP2A((event.x,event.y))

        zoomx = False
        zoomy = False
        if(ya < 0):
            if(xa >= 0 and xa <= 1):
                zoomx = True
                zoomy = False
        elif(ya <= 1):
            if(xa <0):
                zoomx = False
                zoomy = True
            elif(xa <= 1):
                zoomx = True
                zoomy = True
            else:
                zoomx = False
                zoomy = True
        else:
            if(xa >=0 and xa <= 1):
                zoomx = True
                zoomy = False

        if(zoomx):
            scale_factorx = scale_factor
        else:
            scale_factorx = 1.
        if(zoomy):
            scale_factory = scale_factor
        else:
            scale_factory = 1.

        new_axlim0, new_axlim1 = (xa*(1 - scale_factor),
                                  xa*(1 - scale_factor) + scale_factor)
        new_aylim0, new_aylim1 = (ya*(1 - scale_factor),
                                  ya*(1 - scale_factor) + scale_factor)

        #now convert axes to data
        new_xlim0,new_ylim0 = tranSclA2D(tranA2D((new_axlim0,new_aylim0)))
        new_xlim1,new_ylim1 = tranSclA2D(tranA2D((new_axlim1,new_aylim1)))

        #and set limits
        self.ax.set_xlim([new_xlim0,new_xlim1])
        self.ax.set_ylim([new_ylim0,new_ylim1])
        self.redraw()

    def onrelease(self, event):
        '''If the mouse wheel is clicked the image is shifted after dragging
        and releasing. Also works in log plots.'''
        if(event.button == 2):
            #transform pixels to Axes
            tranP2A = self.ax.transAxes.inverted().transform
            #transform Axes to data
            tranA2D = self.ax.transLimits.inverted().transform
            #transform scale of Axes to data
            tranSclA2D = self.ax.transScale.inverted().transform

            #get current position in axes coordinates
            xa,ya = tranP2A((event.x,event.y))

            shiftx = False
            shifty = False
            if(ya < 0):
                if(xa >= 0 and xa <= 1):
                    shiftx = True
                    shifty = False
            elif(ya <= 1):
                if(xa <0):
                    shiftx = False
                    shifty = True
                elif(xa <= 1):
                    shiftx = True
                    shifty = True
                else:
                    shiftx = False
                    shifty = True
            else:
                if(xa >=0 and xa <= 1):
                    shiftx = True
                    shifty = False
    
            dxa,dya = (xa-self.xa0),(ya-self.ya0)
            new_xlima0,new_ylima0 = (-dxa,-dya)
            new_xlima1,new_ylima1 = (1-dxa,1-dya)

            new_xlim0,new_ylim0 = tranSclA2D(tranA2D((new_xlima0,new_ylima0)))
            new_xlim1,new_ylim1 = tranSclA2D(tranA2D((new_xlima1,new_ylima1)))

            if(shiftx):
                self.ax.set_xlim(new_xlim0,new_xlim1)
            if(shifty):
                self.ax.set_ylim(new_ylim0,new_ylim1)

            #update xa0, ya0
            self.xa0,self.ya0 = (xa,ya)
            self.redraw()

    def redraw(self):
        '''Redraw in a quicker manner.''' 
        #Omitted because of memory leaks...
#        imgs = self.ax.images
#        lines = self.ax.lines
#        self.ax.draw_artist(self.ax.patch)
#        if(len(imgs) > 0):
#            for img in imgs:
#                self.ax.draw_artist(img)
#        for collection in self.ax.collections:
#            self.ax.draw_artist(collection)
#        if(len(lines) > 0):
#            for line in lines:
#                self.ax.draw_artist(line)
#        self.ax.figure.canvas.blit(self.ax.bbox)
#        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def onclick(self, event):
        ''' On click, save the x y coordinates for future use for the drag and
        release operation.'''
        self.x0,self.y0 = event.x,event.y
        tranP2A = self.ax.transAxes.inverted().transform
        self.xa0,self.ya0 = tranP2A((self.x0, self.y0))
