
from bisect import bisect_left
import numpy as np
import math

def clamp(val, low, high):
    if val < low:
        return low 
    elif val>high:
        return high 
    else:
        return val
def FindInterval(a, x):
    first = 0 
    size = len(a)
    length = size
    while(length >0):
        half = int(length/2)
        middle = first + half 
        if(a[middle]<=x):
            first = middle + 1
            length -= half+1
        else:
            length = half 
    return clamp(first -1, 0, size-2)

def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i!=len(a) and a[i]==x:
        return i 
    else:
        return -1

class Distribution1D:
    def __init__(self, f, n) -> None:
        self.func = f
        self.cdf = [0]*(n+1)
        for i in range(1, n+1):
            self.cdf[i] = self.cdf[i-1] + self.func[i-1]/n 
        self.funcInt = self.cdf[n]
        if(self.funcInt == 0):
            for i in range(1, n+1):
                self.cdf[i] = float(i)/float(n) 
        else:
            for i in range(1, n+1):
                self.cdf[i] /= float(self.funcInt)
    
    def Count(self):
        return len(self.func)

    def SampleContinuous(self, u):
        offset = FindInterval(self.cdf, u)
        
        du = float(u - self.cdf[offset])
        if(self.cdf[offset+1] - self.cdf[offset] > 0):
            assert(self.cdf[offset + 1] > self.cdf[offset])
            du /= (self.cdf[offset+1] - self.cdf[offset])
        assert(math.isnan(du) == False)

        res =  (offset + du) / self.Count()
        
        if(self.funcInt>0):
            return self.func[offset]/self.funcInt, res , offset
        else:
            return 0, res , offset



class Distribution2D:
    def __init__(self, data, nu, nv) -> None:
        self.pConditionalV = list()
        self.marginalFunc = list()
        for v in range(nv):
            self.pConditionalV.append(Distribution1D(data[v], nu))
        for v in range(nv):
            self.marginalFunc.append(self.pConditionalV[v].funcInt)
        self.pMarginal = Distribution1D(self.marginalFunc, nv)
        
    def SampleContinuous(self, u):
        
        pdfs1, d1, v = self.pMarginal.SampleContinuous(u[1])

        pdfs0, d0, _ = self.pConditionalV[v].SampleContinuous(u[0])
        pdf = pdfs0 * pdfs1
        # return [d0,d1], pdf
        return [d1, d0], pdf #TODO careful
        