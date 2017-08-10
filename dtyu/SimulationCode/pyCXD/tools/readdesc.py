from numpy import loadtxt

def readdesc(filename):
    ''' read a file name descriptor of format
        name = value
        where comments are #
    '''
    myvars = {}
    with open(filename) as myfile:
        for line in myfile:
            if(line[0] != "#"):
                name, var = line.partition("=")[::2]
                myvars[name.strip()] = float(var)
    return myvars
