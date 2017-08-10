''' These are for 3D projected shapes.
    In this case, the shapes can be manipulated in 3 dimensions.
    They come with a few routines to rotate them etc.
'''
import numpy as np
from tools.transforms import rotation_matrix

class Shape3:
    ''' 3D shape class.
        It is made up of types and units. Types are generators of shapes and
        units are the positions of these shapes.
        This makes it possible to quickly generate a structure (such as a crystal)
        of the same shape, while allowing the possibility of constructing a hybrid
        crystal of different shapes (i.e. spheres of different sizes/density).
    '''

    def __init__(self,dims=[1000,1000,1000]):
        ''' Initialize object. 
            dimensions are the dimensions of the object.
            Note: the object is not initialized in memory until
                asked for.
            The default is [1000,1000,1000] but this could change in the future.
        '''
        # The positions and type of each unit
        self.vecs = np.array([],dtype=float).reshape(0,3)
        self.types = np.array([],dtype=int)
        self.dims = dims

        # number of subunits
        self.notypes = 0

        # the data representing each subunit
        self.typefns = []
        self.typeparams = []
        self.typebboxes = []

        # some stuff specifying images
        # right now all these are for the projections
        self.img = None
        self.fft2 = None

        # temporary stuff
        self.curtype = -1
        self.typeimg = None

    def addtype(self, shapefn, shapeparams, bboxdims=None):
        ''' Add another shape type. Returns the number of this type.'''
        self.notypes += 1
        self.typefns.append(shapefn)
        self.typeparams.append(shapeparams)
        self.typebboxes.append(bboxdims)
    
    def addunits(self, vecs, typeno=None):
        ''' Add a unit
            vecs : 3D position vectors (can also just be one vector)
            typeno : the type number of the unit
            Note : if no type specified, it will find last defined type.
                Also note that this is not the type of the last added unit,
                this is the last type that was defined.
            types are 0 based indexed. notypes is always max(types)+1
            2. This is slow. It is assumed that adding vecs is not something
                done often. It will recreate a numpy array for the vectors each time.
                If you need to speed this up, use a buffer (double memory usage
                every time it fills up)
        '''
        if(self.notypes == 0):
            raise ValueError("No types have been defined yet.")
        if(typeno is None):
            typeno = self.notypes-1
        if(typeno > self.notypes-1):
            raise ValueError("The type specified does not exist")
        if(len(vecs) == 3):
            vecs = [vecs]
        elif(len(vecs) == 1):
            if(len(vecs[0]) != 3):
                raise ValueError("The vector is not a list of 3 elements or a list of lists of 3 elements")
        vecs = np.array(vecs)
        typenos = np.tile(typeno,vecs.shape[0])

        # the slow piece
        self.vecs = np.concatenate((self.vecs,vecs))
        self.types = np.concatenate((self.types,typenos))
        # reorder in terms of types (will help make projection faster)
        vecsorder = np.argsort(self.types)
        self.vecs = self.vecs[vecsorder]
        self.types = self.types[vecsorder]

    def project(self):
        ''' Project the units onto a 2D image. The convention
            is to project onto the x-y plane.'''
        if self.img is None:
            self.img = np.zeros((self.dims[0],self.dims[1]))
        else:
            self.clearimg(self.img)
        curtype = -1
        for vec, typeno in zip(self.vecs, self.types):
            if typeno != curtype:
                # first clear old shape
                if(self.typeimg is None):
                    self.typeimg = np.zeros((self.dims[0],self.dims[1]))
                if curtype >= 0:
                    self.clearimg(self.typeimg, bboxdims=self.typebboxes[curtype])
                curtype = typeno
                # make a new type
                self.gentype(typeno)
            # project vector onto z, round, project current type stored
            self.projecttype((vec[:2]+.5).astype(int))
        self.fimg2 = np.fft.fftshift(np.abs(np.fft.fft2(self.img)))**2

    def transform3D(self, tranmat):
        ''' Transform the 3D vectors according the transformation 
            matrix.'''
        #tvecs = np.array(vecs)*0;
        self.vecs = np.tensordot(self.vecs,tranmat,axes=(1,1))
        
    def gentype(self, typeno):
        ''' Make the temporary type specified by the typeno.'''
        curbbox = self.typebboxes[typeno]
        # project sphere onto the image
        self.typefns[typeno](self.typeimg,*(self.typeparams[typeno]),bboxdims=curbbox)
       
    def projecttype(self, r): 
        ''' Project the current type in the class to the image at the position specified.
            bbox indexing is left biased: [-bd[0]//2, (bd[0]-1)//2]
        '''
        bd = self.typebboxes[self.curtype]
        # xleft, yleft, xright, yright
        bboxtype = [0, 0, bd[0]-1, bd[1]-1]
        bboximg = [r[0] - bd[0]//2, r[1] - bd[1]//2, r[0] + (bd[0]-1)//2, r[1] + (bd[1]-1)//2]
        #bounds check (need to speed this up later)

        # x check
        if(bboximg[0] < 0):
            bboxtype[0] -= bboximg[0]
            bboximg[0] = 0
            
        if(bboximg[0] >= self.img.shape[1]):
            bboxtype[0] -= (bboximg[0] - (self.img.shape[1]-1))
            bboximg[0] = self.img.dims[1] - 1

        # y check
        if(bboximg[1] < 0):
            bboxtype[1] -= bboximg[1]
            bboximg[1] = 0
            
        if(bboximg[1] >= self.img.shape[0]):
            bboxtype[1] -= (bboximg[1] - (self.img.shape[0]-1))
            bboximg[1] = self.img.dims[0] - 1
        
        if(bboximg[0] != bboximg[2] and bboximg[1] != bboximg[3]):
            self.img[bboximg[1]:bboximg[3]+1,bboximg[0]:bboximg[2]+1] += self.typeimg[bboxtype[1]:bboxtype[3]+1,bboxtype[0]:bboxtype[2]+1]


    def plotbbox(self, vecno):
        ''' Plot an image with the bounding box of the vector highlighted
            based on the vector number vecno
            Not implemented but the aim is to find which element a vector designates.
            Will project in 2D.
        '''
        pass

    def clearimg(self,img,bboxdims=None):
        ''' Clear the last projected shape.
            Note: setting bboxdims speeds things up.
            Set bboxdims = None to clear the whole image.
            This routine will eventually be written in cython.
        '''
        if(bboxdims is None):
            bboxdims = img.shape
        img[:bboxdims[0], :bboxdims[1]] = 0

# Some shape generating functions, will eventually be written in cython
def sphereprojfn(img, r, rho, alpha, beta, gamma, bboxdims=None):
    ''' Draw the projection of a sphere.
        will be drawn in center of bounding box
        r - radius
        rho - density 
        img - image to project to

        Bounding box details: for bounding boxes with even number of dimensions, the central 
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.
    '''
    if(bboxdims is None):
        bboxdims = img.shape

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    rcen = (bd[0]-1)//2, (bd[1]-1)//2;

    x = np.arange(bd[0])
    y = np.arange(bd[1])
    X,Y = np.meshgrid(x,y)
    img[:bd[1],:bd[0]] = rho*np.sqrt(np.maximum(0,(1- ((X-rcen[0])**2 + (Y-rcen[1])**2)/r**2)))


# TODO : grab arguments of function and return a dictionaryfor these arguments (see how lmfit.Model does it)
# TODO : make command line prompt to ask for values of function arguments
