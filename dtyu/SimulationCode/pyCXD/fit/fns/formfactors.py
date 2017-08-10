import numpy as np
from math import factorial
#Bessel function of the first kind
from scipy.special import jv
#If this gets too big, move to separate files
#stealing from http://iramis.cea.fr/en/Phocea/Vie_des_labos/Ast/ast_sstechnique.php?id_ast=1799
from scipy.interpolate import RectBivariateSpline

def sqsphereschultz(q,pars):
    ''' S(q) of a sphere with a polydispersity according to the Schultz 
        distribution.
        amp : amplitude at q=0
        r1 : the radius r1 (avg radius)
        Z :  the Schultz factor, related to sigmar by:
                sigmar = r1/sqrt(z+1)
                (so z = (r1/sigmar)**2 - 1 )
        bg : a constant background
        parasitic : parasitic scattering (1/q^4)
        
        From formula defined in my logbook, log #0066
        Orig formula taken from 
            The Journal of Chemical Physics 79(5):2461-2469, 1983
        '''
    if(hasattr(pars['rbar'],"value")):
        rbar = pars['rbar'].value
        z = pars['z'].value
        amp = pars['amp'].value
        parasitic = pars['parasitic'].value
        bg = pars['bg'].value
    else:
        rbar = pars['rbar']
        z = pars['z']
        amp = pars['amp']
        parasitic = pars['parasitic']
        bg = pars['bg']

    alphap = q*rbar/(z+1)
    th = np.arctan(2*alphap)
    g1 = 1.
    g1 -= (4*alphap**2+1)**(-.5*(z+1))*np.cos((z+1)*th)
    g1 += (z+2)*(z+1)*alphap**2*(1 + (4*alphap**2+1)**(-.5*(z+3))*np.cos((z+3)*th))
    g1 -= 2*(z+1)*alphap*(4*alphap**2 + 1)**(-.5*(z+2))*np.sin((z+2)*th)
    g1 /= (q*rbar)**6
    return amp*g1 + parasitic/q**4 + bg


def spheresqfunc(q,pars):
    '''Sphere form factor.
        pars[0] - amplitude
        pars[1] - radius R
        S(qR)
        '''
    if(hasattr(pars['radius'],"value")):
        qR = q*pars['radius'].value
        amp = pars['amp'].value
    else:
        qR = q*pars['radius']
        amp = pars['amp']
    return (pars['amp']*3*(np.sin(qR) - qR*np.cos(qR))/qR**3)**2

def sspheresqfunc(q,pars):
    '''Sphere form factor smoothed. This mimicks polydispersity in a crude fashion.
        pars[0] - amplitude
        pars[1] - radius R
        pars[2] - spread over qR domain (assumed uniform)
    '''
    if(hasattr(pars['window'])):
        window = pars['window'].value
    else:
        window = pars['window']
    return savitzky_golay(spheresqfunc(q,pars),window,1)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    '''Savitzky golay filter, from 
        http://stackoverflow.com/questions/22988882/how-to-smooth-a-curve-in-python
    '''
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except (ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def spherepolygauss(q,pars):
    """
    #ref
    Taken from http://iramis.cea.fr/en/Phocea/Vie_des_labos/Ast/ast_sstechnique.php
    q - scattering vector (invs angs)
    pars['amp'] Amplitude
    pars['radius'] Mean radius(A)
    pars['FWHM'] FWHM (A)
    pars['parasitic'] parasitic (Porod's law) scattering
    pars['bg'] background scattering
    """
    if(hasattr(pars['radius'],"value")):
        a=pars['radius'].value
        s=pars['FWHM'].value/(2.*a*(np.log(2.))**0.5 )
        parasitic = pars['parasitic'].value
        background = pars['bg'].value
        amp = pars['amp'].value
    else:
        a=pars['radius']
        s=pars['FWHM']/(2.*a*(np.log(2.))**0.5 )
        parasitic = pars['parasitic']
        background = pars['bg']
        amp = pars['amp']

    t1=(q*a*s).astype(float)
    t2=(2.*q*a).astype(float)
    t3=(q*a).astype(float)
    f1=1.+ (1.+0.5*s**2.)*((t3)**2.) - (t2)*( 1.+(t1**2.))*np.sin(t2)*np.exp(-t1**2.) - ( 1.+(1.5*(s**2.)-1.)*(t3**2.) + (t1**4.) )*np.cos(t2)*np.exp(-(t1**2.))
    f2=4.5*(t3**(-6.))*(1.+7.5*s**2.+(45./4.)*s**4.+(15./8.)*s**6.)**(-1)
    #normfactor=((4.*(a**3.)*np.pi/3.)**2.)*(1.+7.5*(s**2.)+(45./4.)*(s**4.)+(15./8.)*(s**6.))
    return amp*f1*f2 + parasitic/q**4 + background

def ring2Dfitfunc(xy,pars):
    '''2D ring fit function, used for calibration.
        amp     : amplitude
        r0      : radius
        sigma   : std dev
        x0      : center x
        y0      : centery
    '''
    if(hasattr(pars['amp'],"value")):
        amp     = pars['amp'].value
        r0      = pars['r0'].value
        sigma   = pars['sigma'].value
        x0      = pars['x0'].value
        y0      = pars['y0'].value
    else:
        amp     = pars['amp']
        r0      = pars['r0']
        sigma   = pars['sigma']
        x0      = pars['x0']
        y0      = pars['y0']

    #xy.reshape((2,xy.shape[0]/2))
    r = np.sqrt((xy[0]-x0)**2 + (xy[1] - y0)**2)
    return amp*np.exp(-(r-r0)**2/2./sigma**2)

#Normalized form factors

def fqsphere(q,pars):
    '''Sphere form factor.
        pars[0] - amplitude
        pars[1] - radius R
        S(qR)
        '''
    if(hasattr(pars['radius'],"value")):
        qR = q*pars['radius'].value
    else:
        qR = q*pars['radius']
    return (3*(np.sin(qR) - qR*np.cos(qR))/qR**3)

def fq2sphere(q,pars):
    '''Sphere form factor.
        pars[0] - amplitude
        pars[1] - radius R
        S(qR)
        '''
    if(hasattr(pars['amp'],"value")):
        amp = pars['amp'].value
    else:
        amp = pars['amp']
    return amp*np.abs(fqsphere(q,pars))**2

def fqdisc(qR):
    '''The |F(q)| for a disc, normalized to 1 at q=0
    amp : the amplitude
    radius : the radius
    '''
    return 2*jv(1,qR)/qR

def fq2disc(q,pars):
    '''The |F(q)|^2 for a disc
    amp : the amplitude
    radius : the radius
    '''
    qR = q*pars['radius']
    return pars['amp']*np.abs(fqdisc(qR))**2

def fqring(q,pars):
    '''The |F(q)| for a ring. We take the product with r*2 because they both scale as area.
    radius - average radius 
    rwidth - width of ring
    '''
    r1 = (pars['radius'] - pars['rwidth']*.5)
    r2 = (pars['radius'] + pars['rwidth']*.5)
    qR1 = q*r1
    qR2 = q*r2
    if(r1 < 0):
        return fqdisc(qR2)
    else:
        return (r2**2*fqdisc(qR2) - r1**2*fqdisc(qR1))/(r2**2-r1**2)

def fq2ring(q,pars):
    '''The |F(q)|^2 for a ring.
    radius - average radius 
    rwidth - width of ring
    amp - amplitude
    '''
    return pars['amp']*np.abs(fqring(q,pars))**2

F4Cinterp = None

def fq2_4circ(Q,pars):
    '''NOTE: This is slow.
        Compute the |Fourier transform|**2 of a quarter circle. This is an analytical
        calculation. It is assumed that not more than a 10 oscilations would
        be needed so circle is made to be 1/10th of image. 1000 x1000 pixels
        are used.
        The interpolation function's domain is in units of radians per pixel for a
        half circle at 100 pixels in radius.
    '''
    global F4Cinterp;
    if(F4Cinterp is None):
        print("F4 interpolation function empty, creating it for the first time...")
        N = 1000
        X,Y = np.mgrid(N,N)
        QCimg = (sqrt(X**2 + Y**2) < 100**2).astype(int)
        FQC = fftshift(fft2(Qimg))
        X = (X-N/2.)*2*np.pi/N
        Y = (Y-N/2.)*2*np.pi/N
        F4Cinterp = RectBivariateSpline(X,Y,FQC)
    radius = pars['radius']
    Q = radius/100.*Q
    F4c = F4Cinterp(Q[0],Q[1])
