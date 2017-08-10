# routines to deal with lists
from detector.eiger import EigerImages

def lopen(listname, n, DDIR=None):
    ''' Open the nth element (0 based indexing) in list with listname
        NOTE : it loads the whole file to find the element. Ideally, it should stop at the nth line
    '''
    if DDIR is None:
        DDIR = ""
    for i, line in enumerate(open(listname)):
        if i == n:
            filename = line.rstrip('\n')
            break
    imgs = EigerImages(DDIR + "/" + filename)
    return imgs

def listlen(listname, DDIR = None):
    ''' Open the nth element (0 based indexing) in list with listname
        NOTE : it loads the whole file to find the element. Ideally, it should stop at the nth line
    '''
    if DDIR is None:
        DDIR = ""
    lines = [line for line in open(listname)]
    return len(lines)
