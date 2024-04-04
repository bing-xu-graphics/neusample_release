
from bisect import bisect_left
import numpy as np
import math
import torch
from sys import exit

M_PI = 3.14159265359
PI_over_4 = 0.78539816339
PI_over_2 = 1.57079632679

torch.set_default_dtype(torch.float64)
NUM_UV_DIM = 512
NUM_WO_VARY = 512
NUM_WI_SAMPLES = 1024
WI_RES = 256
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
        
"""
sample wi using the optimal pdf 
given wo and uv
"""
def importance_sample_wi(uv_wo, img,  num_samples):
        samples = list()
        resolution = len(img)
        distribution = Distribution2D(img, resolution, resolution)
        for i in range(num_samples):
            sampleWi, pdf = distribution.SampleContinuous(torch.rand(2))
            sampleWi[0] = sampleWi[0]*2.0 - 1.0
            sampleWi[1] = sampleWi[1]*2.0 -1.0
            if sampleWi[0]* sampleWi[0] + sampleWi[1] * sampleWi[1] > 1.0:
                print("one invalid")
            samples.append([*uv_wo, *sampleWi])
        return samples

class OptimalSampler():
    def __init__(self):
        ### wi grids
        wiX = torch.linspace(-1.0,1.0, steps = WI_RES)
        wiY = torch.linspace(-1.0,1.0, steps = WI_RES)
        grid_z1, grid_z2 = torch.meshgrid(wiX, wiY)
        gridwi = torch.stack([grid_z1, grid_z2], dim = -1)
        light_dir = gridwi.reshape((-1, 2))
        invalid_dirs = torch.square(light_dir[...,0]) + torch.square(light_dir[...,1]) > 0.995 #TODO CHANGED here
        self.light_dir = light_dir
        self.invalid_dirs = invalid_dirs
        self.A = 4/(WI_RES * WI_RES)

    """given uv and camera_dir, sample the light_dir, num_samples is usually just 1 in path tracing"""
    def sample(self, uv, camera_dir, num_samples):
        uv_wo_records = torch.cat((uv, camera_dir), dim=-1)
        light_dir_tensor = self.light_dir.cuda()

        ## tabulating descritized wi
        uv_tensor = torch.tile(uv, (light_dir_tensor.shape[0],)).reshape(-1,2).cuda()
        camera_dir_tensor = torch.tile(camera_dir, (light_dir_tensor.shape[0],)).reshape(-1,2).cuda()
        with torch.no_grad():
            rgb_pred = self.renderer.model(camera_dir_tensor, light_dir_tensor, uv_tensor) #uv_tensor

            wi_img = rgb_pred.cpu()
            wi_img[self.invalid_dirs] = 0.0
            lumi_pred = self.renderer.rgb2lum(wi_img)
            lumi_pred *= 1/(self.A * torch.sum(lumi_pred[:,0])) #no need for normalization for sampling
            lumi_pred = lumi_pred.reshape(WI_RES, WI_RES, -1)#.cpu()
            gt_pdf = lumi_pred[:,:,0]
            wi_samples = self.importance_sample_wi(uv_wo_records, gt_pdf, num_samples)
            return wi_samples
