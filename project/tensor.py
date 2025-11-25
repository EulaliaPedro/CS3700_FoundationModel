import numpy as np
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


class tensor1D:
    def __init__(self, kernel_tensor, width):
        # Like the this in java (this.name)
        self.kernel_tensor = kernel_tensor
        self.width = width

    # ----------------- Initilize CUDA ----------------- #
        # Determines the number of GPUs
        self.num_gpus = driver.Device.count()

        # Checking the number of GPUs
        #print("Number of GPUs detected:", self.num_gpus)
        #if self.num_gpus == 0:
         #  raise RuntimeError("No GPUs found!")

    def run(self, W, x, b):

        # Split the rows across GPUs
        split_W = np.array_split(W, self.num_gpus, axis=0)
        split_b = np.array_split(b, self.num_gpus, axis=0)

        # Stores the results for the GPU
        split_results = []

    # ----------------- For GPU ----------------- #
        for i in range (self.num_gpus):
            #create a context for this GPU
            context = driver.Device(i).make_context()

            try:
        # Convert matrix into float32 and flatten into 1D array
                W_slice = split_W[i].astype(np.float32)
                b_slice = split_b[i].astype(np.float32).ravel()
                x_full = x.astype(np.float32).ravel()

                # Reshape the matrix that was split
                rows = W_slice.shape[0]
                cols = W_slice.shape[1]

                # Transfer data to GPU
                device_W = gpuarray.to_gpu(W_slice.ravel())
                device_x = gpuarray.to_gpu(x_full)
                device_b = gpuarray.to_gpu(b_slice)

                # An empty GPU array to store the result
                device_y = gpuarray.empty(rows, dtype=np.float32)

                # Compile kernel_code
                mod = SourceModule(self.kernel_tensor)
                kernel = mod.get_function("matrix_vec")

                # Define the sizes
                block = (256, 1, 1)
                grid = (rows, 1)

                # Launch the kernel
                kernel(device_W, device_x, device_b, device_y,
                   np.int32(self.width),
                   block=block,
                   grid=grid
                )

                # Collect and append results into the list
                split_results.append(device_y.get())

            # Clean up
            finally:
                context.pop()
                context.detach()
        # Concatenate into a 1D array
        return np.concatenate(split_results, axis=0)


