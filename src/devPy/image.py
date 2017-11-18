'''
Created on Nov 16, 2017

@author: rykath
'''

from astropy.io import fits

def writeFits(data,info):
    # data is numpy array
    hdu = fits.PrimaryHDU(data)
    hdu.header['class'] = 'Mandelbrot'
    hdu.header['axis-X'] = 'C.imag'
    hdu.header['axis-Y'] = 'C.real'
    hdu.header['values'] = 'escape time'
    
    hdu.header['formula'] = 'Z[0] = K ; Z[n+1] = Z[n]^2 + C'
    hdu.header['cen-C'] = info['cenC']
    hdu.header['cen-K'] = 0+0j
    hdu.header['scale'] = info['scale']
    hdu.header['max-iter'] = info['maxI']
    
    hdu.header['prog-by'] = 'rykath'
    hdu.header['lang'] = 'python'
    hdu.header['optim'] = 'none'
    hdu.header['calc-by'] = 'rykath'
    hdulist = fits.HDUList([hdu])
    hdulist.writeto('test.fits',clobber = True)
