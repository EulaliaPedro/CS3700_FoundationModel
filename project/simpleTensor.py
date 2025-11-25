import numpy as np
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


driver.init()

class SimpleTensor:

     # Initialize
     def __init__(self, kernel_tensor, WIDTH):
         self.kernel_tensor = kernel_tensor
         self.WIDTH = WIDTH

    # ----------------- Initilize CUDA ----------------- #
         #determines the number of gpus
         self.num_gpus =  driver.Device.count()


     def run(self, W, x, b):
         # W: matrix
         # x: input vector
         # b: bias vector

         # Splits the matrix and bias vector for each GPU
         split_W = np.array_split(W, self.num_gpus, axis=0)
         split_b = np.array_split(b, self.num_gpus)

         # Stores the results of the GPU
         results = []

     # ----------------- For GPU ----------------- #
         for i in range(self.num_gpus):
             #create context
             context = driver.Device(i).make_context()
             try:
                 # Transfer data to GPU
                 device_W = gpuarray.to_gpu(split_W[i].ravel().astype(np.float32))
                 device_x = gpuarray.to_gpu(x.ravel().astype(np.float32))
                 device_b = gpuarray.to_gpu(split_b[i].ravel().astype(np.float32))

                 # Assign the spilt matrix into the current GPU
                 rows = split_W[i].shape[0]

                 # An empty GPU array to store the result
                 device_y = gpuarray.empty(rows, dtype=np.float32)

                 # Compile kernel_code
                 mod = SourceModule(self.kernel_tensor)
                 kernel = mod.get_function("matrix_vec")

                 # Define the sizes
                 block = (1, 1, 1)
                 grid = (rows, 1)

                 # Launch the kernel
                 kernel(device_W, device_x, device_b, device_y,
                        np.int32(self.WIDTH),
                        block = block, grid = grid
                 )

                 # Collect and append results into the list
                 results.append(device_y.get())

             finally:
                 # Clean up
                 context.pop()
                 context.detach()

         # Concatenate into a 1D array
         return np.concatenate(results)
