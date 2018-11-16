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


def do_ffts(img):

    if img.dtype != 'float32':
        img = img.astype('float32')

    sx, sy = img.shape # or the convention (width x height) vs (rows x columns)
    # is sure to bite you
    
    ## Prepare and run CUDA FFT on image
    # See https://www.idtools.com.au/gpu-accelerated-fft-compatible-numpy/
    time0=time.time()
    
    # Initialise CUDA input GPUArray
    x_gpu = gpuarray.to_gpu(img)

    # Initialise output GPUarray
    # N/2+1 non-redundant coefficients of a length-N input signal
    y_gpu = gpuarray.empty((sx,sy//2+1), np.complex64)

    # Plan and run Cuda fft
    plan_fft = cu_fft.Plan((sx, sy), np.float32, np.complex64)
    cu_fft.fft(x_gpu, y_gpu, plan_fft)
    gpu_fft = y_gpu.get()
    gpu_time = time.time() - time0

    print(f'GPU FFT preparation, execution and retrieval in {gpu_time:6.4f} s')

    ## Run np fft
    time0 = time.time()
    cpu_fft = np.fft.fft2(img)
    cpu_time = time.time() - time0
    print(f'NumPY CPU FFT in {cpu_time:6.4f} s')

    return(gpu_fft, cpu_fft, gpu_time, cpu_time)


## Create 2D array
im=Image.open("/home/fredrik/Pictures/bg2.jpg")
img=im.convert("L") # luminance
im.close()


## Compare cuda and np fft for different scalings
sizes = [ (128, 64), (256, 128), (512, 256), (1024, 512), (2048, 1024), (4096, 2048), (8192, 4096), (16384, 8192) ]
gtime=[]
ctime=[]

for size in sizes:

    # Scale the original image
    scimg=img.resize(size)
    scimg=np.array(scimg)
    
    ax=plt.subplot(222)
    ax.set(title=f'Scaled image {size[0]} x {size[1]}')
    plt.imshow(scimg,cmap='gray')
    
    # Run FFT operations
    gpu_fft, cpu_fft, gpu_time, cpu_time = do_ffts(scimg)
    gtime.append(gpu_time)
    ctime.append(cpu_time)
    
    #np.fft compatibility: stack the array and its flipped version
    flip=gpu_fft.copy()
    flop=np.roll(np.fliplr(np.flipud(flip))[:,1:-1],1,axis=0)
    gpu_fft = np.fft.fftshift(np.hstack((flip, flop)))

    cpu_fft = np.fft.fftshift(cpu_fft)

    ax=plt.subplot(221)
    ax.set(title="CUDA FFT")
    plt.imshow(np.abs(gpu_fft).astype('uint8'),cmap='gray')

    ax=plt.subplot(223)
    ax.set(title="CPU FFT")    
    plt.imshow(np.abs(cpu_fft).astype('uint8'),cmap='gray')

    ax=plt.subplot(224)
    ax.set(title="Difference relative to CPU FFT")
    plt.imshow( (100.0*(np.abs(gpu_fft)/np.abs(cpu_fft) - 1.0)).astype('uint8'),cmap='plasma')
    plt.colorbar(label='%')
    
    # plt.show()
    plt.savefig(f'ffts_{size[0]}.pdf')
    plt.close()

w=[s[0] for s in sizes]

ax=plt.subplot(111)

ax.plot(w, gtime, 'r.-', w, ctime, 'b.-')
ax.set(title='CPU and GPU calculation times', xscale='log', yscale='log', xlabel='Image width', ylabel='s' )
ax.legend(('GPU','CPU'))
ax.grid(True,which='both')

# plt.show()
plt.savefig('run_times.pdf')
plt.close()
