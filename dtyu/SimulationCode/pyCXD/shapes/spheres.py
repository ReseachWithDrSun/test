'''Creation of spheres (3D).'''

def generatesphere(R,pars):
    '''Generate spheres centered at (pars[2],pars[3],pars[4]). R is an array of
    [x,y,z] positions.
        pars[0] - amplitude
        pars[1] - radius
        pars[2] - xcen
        pars[3] - ycen
        pars[4] - zcen
        '''
    return (pars[0]*((R[0,:] - pars[2])**2 + (R[1,:] - pars[3])**2 + (R[2,:]-pars[4])**2 < pars[1]**2))
