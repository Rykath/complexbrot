'''
Created on Nov 18, 2017

@author: rykath
'''

import numpy

def mandelSequence(C,Z,maxI):
    for i in range(maxI):
        if abs(Z) > 2:
            return i
        Z = Z**2 + C
    return maxI

def mandelbrotSimple(cenC,scale,samples,maxI):
    data = numpy.empty(samples)
    it = numpy.nditer(data, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        it[0] = mandelSequence(cenC+scale*complex(it.multi_index[1]-samples[1]/2,it.multi_index[0]-samples[0]/2),0,maxI)
        it.iternext()
    return data
            
