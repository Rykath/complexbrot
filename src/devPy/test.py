'''
Created on Nov 18, 2017

@author: rykath
'''

if __name__ == '__main__':
    import devPy.image
    import devPy.brot
    
    info = {}
    info['cenC'] = -0.5+0j
    info['samples'] = [400,400]
    info['scale'] = 3.0 / info['samples'][0]
    info['maxI'] = 100
    
    data = devPy.brot.mandelbrotSimple(info['cenC'],info['scale'],info['samples'],info['maxI'])
    
    devPy.image.writeFits(data,info)