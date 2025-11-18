import numpy as np
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


#Kenerel for the matrix-vector multiplication
kernel_code = """
__global__ void dot_rows(float *W, float *x, float *b, float *result, int WIDTH) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= WIDTH||col>=WIDTH) return;

    int idx = row * WIDTH + col;

    //multiply corresponding elements and add the bias
    result[idx] = W[idx] * x[col] + b[col];
}
"""


# ----------------- Initilize CUDA ----------------- #
driver.init()
num_gpus = driver.Device.count() #detects the number of GPUs
print("Number of GPUs detected:", num_gpus)
if num_gpus == 0:
    raise RuntimeError("No GPUs found!")

# ----------------- FROM LAB 5 -----------------
WIDTH = 6

#Random square matrix (W)
host_W = np.random.randint(1,10, size=(WIDTH, WIDTH)).astype(np.float32)

#Random column vector (x)
host_x = np.random.randint(1,10, size=(WIDTH, 1)).astype(np.float32)

#Random bias vecotr (b)
host_b = np.random.randint(1,5,size=(1, WIDTH)).astype(np.float32)

#split the rows across GPUs
split_W = np.array_split(host_W, num_gpus, axis=0)

split_results = [] #empty list, will store the partial results

# Create contexts for each GPU (inactive)
for i in range(num_gpus):

    #create a CUDA context for this GPU
    ctx = driver.Device(i).make_context()

    try:
        #Slicing the matrix(W) for this GPU
        W_slice = split_W[i].astype(np.float32)
        rows = W_slice.shape[0]

        # Print the slice for this GPU
        print("\n--- GPU {} will process these rows ---\n{}\n".format(i, W_slice))

        #Transfer data to GPU
        device_W = gpuarray.to_gpu(W_slice.ravel())
        device_x = gpuarray.to_gpu(host_x.ravel())
        device_b = gpuarray.to_gpu(host_b.ravel())

        # Allocate result array on GPU
        device_result = gpuarray.empty(rows * WIDTH, dtype=np.float32)

        # Compile kernel in current GPU
        mod = SourceModule(kernel_code)
        dot_rows = mod.get_function("dot_rows")

        # Defininf the size of block and grid
        block = (WIDTH, 1, 1)
        grid = (rows, 1)

        #Launching kernel
        dot_rows(
                device_W, device_x, device_b, device_result,
                np.int32(WIDTH), #np.int32(rows),
                block=block, #_x),, 1, 1),
                grid=grid #_x, 1))
        )

        #compying the result of the GPU back into host and reshape the matrix
        split_results.append(device_result.get().reshape(rows, WIDTH))
    finally:  #clean the CPU content
        ctx.pop()
        ctx.detach()

# Combine the results from the GPUs
final_result = np.vstack(split_results)

# ---------- CPU Computation ---------- #
cpu_result= host_W * host_x.ravel() + host_b  # multiplies each column by x[j]

# ---------------- PRINT OUTPUT (testing for now)---------------- #
print("\nHost W:\n", host_W)
print("\nHost x:\n", host_x)
print("\nHost b:\n", host_b)

# ---------------- OUTPUT Result ---------------- #
print("\nMatrix Result from GPU: ")
print(final_result)

# ---------------- OUTPUT Result (test) ---------------- #
#checking result on CPU
print("\nResult from CPU:")
print(cpu_result)

#if it matches in both the CPU and GPU
print("\nMatch:", np.allclose(final_result, cpu_result))













