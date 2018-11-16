#! /usr/bin/optirun python3
# Run Python3 with NVidia GPU available

### 2D FFT of image

## Libraries
import time
import numpy as np
import pycuda.autoinit #PyCUDA
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft #Scikit-CUDA
import matplotlib.pyplot as plt
from PIL import Image

## Create 2D array
im=Image.open("/home/fredrik/Pictures/bg2.jpg")
im=im.convert("L") # luminance
im=im.resize([4096,2048])
img=np.array(im)
im.close()

ax=plt.subplot(311)
plt.imshow(img)
ax.set(title="Scaled image 4096x2048")


if img.dtype != 'float32':
    img = img.astype('float32')

## Prepare and run CUDA FFT on image
# See https://www.idtools.com.au/gpu-accelerated-fft-compatible-numpy/
time0=time.time()

# Initialise CUDA input GPUArray
x_gpu = gpuarray.to_gpu(img)
    
# Initialise output GPUarray
# N/2+1 non-redundant coefficients of a length-N input signal.
y_gpu = gpuarray.empty((2048,2049), np.complex64)

# Plan and run Cuda fft
plan_fft = cu_fft.Plan((2048, 4096), np.float32, np.complex64)
cu_fft.fft(x_gpu, y_gpu, plan_fft)

# np.fft compatibility: stack horizontally the y.get() array and its flipped version
left = y_gpu.get()
right = np.roll(np.fliplr(np.flipud(y_gpu.get()))[:,1:-1],1,axis=0)
gpu_fft = np.fft.fftshift(np.hstack((left,right)))

print("GPU FFT preparation, execution and postprocessing in ", time.time() -time0)

## Run np fft
time0=time.time()
cpu_fft=np.fft.fftshift(np.fft.fft2(img))
print("NumPY CPU FFT in ", time.time() - time0)


## Compare cuda and np fft
ax=plt.subplot(312)
plt.imshow(np.abs(gpu_fft))
ax.set(title="CUDA FFT")

ax=plt.subplot(313)
plt.imshow(np.abs(cpu_fft))
ax.set(title="CPU FFT")

plt.show()
