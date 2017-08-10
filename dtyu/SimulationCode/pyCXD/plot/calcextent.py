def calcextent(x,y):
    '''Calculate the extent of an image based on the two arrays x and y
    '''
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    xlim1 = [x[0]-.5*dx,x[-1]+.5*dx]
    ylim1 = [y[0]-.5*dy,y[-1]+.5*dy]
    return [xlim1[0],xlim1[1],ylim1[1],ylim1[0]]
