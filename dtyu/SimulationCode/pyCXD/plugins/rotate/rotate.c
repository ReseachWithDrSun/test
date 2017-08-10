#include "rotate.h"
int rotate(double *img1, double *img2, double theta, double cenx, double ceny, int dimx, int dimy){
    /*
     *  Rotate img1 into img2 by theta. This routine is meant to be fast.
     *  Only necessary pixels are rotated.
     *  X, Y : the original X,Y coordinaes
     *  XP,YP, arrays with new coordinates (just used as space)
     *  theta : angle to rotate by
     *  dimx,dimy : the image width and height
     *  pixels : list of pixel indexes to rotate
     *  nopixels : the length of the pixel list
     * Could use lib CBLAS but avoiding as much as possible
     *
     */
    double cth, sth;
    int xr, yr;
    int i, j;
    cth = cos(theta);
    sth = sin(theta);
    for(i = 0; i < dimx; i++){
        for(j = 0; j < dimy; j++){
	        //printf("%d\n",pixels[i]);
            //to round: (int)(a + .5)
	        xr = (int)((i-cenx)*cth - (j-ceny)*sth + cenx + .5);
	        yr = (int)((i-cenx)*sth + (j-ceny)*cth + ceny + .5);
	        //rotate by rotating source coordinates, not destination
	        // the latter would lead to missing points
	        if((xr >= 0) && (xr < dimx) && (yr >= 0) && (yr < dimy)){
	            img2[i + dimx*j] = img1[xr + dimx*yr];
	        }
	        //printf("i: %d\n",i);
        }
    }

    return 1;
}

void zero_elems(double *arr, int nopixels){
    /*
 *  Quickly zero an array at pixels indices, meant to use with rotate and meant
 *  to be fast.
     */
    int i;
    for(i = 0; i < nopixels; i++){
        arr[i] = 0;
    }
}
