from scipy.ndimage.interpolation import shift

def dirderivative(img,dirvec):
    '''Take a quick directional derivative with direction dirvec.
        This is a lazy way of doing it, wraps around the edge.
    '''
    dx = img - shift(img,[0,1])
    dy = img - shift(img,[1,0])
    return dx*dirvec[0] + dy*dirvec[1];

