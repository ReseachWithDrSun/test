#Scientific logging
import matplotlib.pyplot as plt

class SciLog:
    def __init__(self,logdir):
        self.logdir = logdir

    def setlogdir(self,logdir):
        self.logdir = logdir

    def sfig(self,name,fig=None,winnum=None):
        if(winnum is not None and fig is None):
            fig = plt.figure(winnum)
        elif(fig is None):
            fig = plt.gcf()
        fig.savefig(self.logdir + "/figs/" + name + ".ps")
