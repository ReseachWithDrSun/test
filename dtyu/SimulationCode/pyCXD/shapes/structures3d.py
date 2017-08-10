''' Making 3 dimensional structures.'''

class Struc3D:
    def __init__(self,positions,elem):
        ''' A 3d structure. Contains vector positions of elements.
            Also contains form of the elements making up the structure'''
        self.positions = positions
        self.elem = elem

class Elem3D:
    '''An element.'''
    def __init__(self,elemfunc, pars):
        self.genelem = elemfunc
        self.pars = pars



