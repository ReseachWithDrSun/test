import matplotlib.pyplot as plt
from tools.transforms import rotation_matrix
from shapes.shapes3DProj import Shape3, sphereprojfn
import numpy as np

shp = Shape3()
# sphere parameters: radius, density, alpha, beta, gamma (Euler angles)
# note for sphere euler angles are not necessary, but this prototype still 
# requires them
shp.addtype(sphereprojfn,[50,2,1,1,1],bboxdims=[100,100]);
shp.addtype(sphereprojfn,[10,2,1,1,1],bboxdims=[100,100]);
shp.addunits([200,200,340],typeno = 0)
shp.addunits([300,200,300],typeno = 1)
shp.addunits([400,200,300],typeno = 0)
shp.project()
img1 = np.copy(shp.img)

# makes a rotation matrix about the [1,1,1] direction
mat = rotation_matrix([1,1,1],5./57.3)
for i in range(100):
    shp.transform3D(mat)
    shp.project()
    print("{}".format(i))
    plt.figure(0);plt.cla();
    plt.imshow(shp.img)
    plt.figure(1);plt.cla();
    plt.imshow(shp.fimg2);
    plt.clim(0,1e5);
    plt.draw();plt.pause(.001)
