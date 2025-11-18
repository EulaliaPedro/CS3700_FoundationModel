import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit


#----------------- FROM LAB 5 -----------------
WIDTH = 3
#creating a square matrix with random variables
host_W = np.random.randint(1,10, size=(WIDTH, WIDTH)).astype(np.float32)

#create a single vector (x)
host_x = np.random.randint(1,10, size=(WIDTH, 1)).astype(np.float32)

#create the bias vecotr (b)
host_b = np.random.randint(1,5,size=(1, WIDTH)).astype(np.float32) 

#transfer host (CPU) memory to device (GPU) memory
device_W = gpuarray.to_gpu(host_W)
device_x = gpuarray.to_gpu(host_x)
device_b = gpuarray.to_gpu(host_b)


# ---------------- PRINT OUTPUT (testing for now)----------------
print("\nDevice W:")
print(device_W)

print("\nDevice x:")
print(device_x)

print("\nDevice b:")
print(device_b)

# ---------------- f_i = Wx_i + b-i ----------------
# Initialize output array on GPU to store the results in a (1, WIDTH) vector 
device_result = gpuarray.zeros((1, WIDTH), dtype=np.float32)

# ---------------- In GPU Perform matrix vector multiplication row by row  ---------------- #
for i in range (WIDTH):
  #device_W[i, ;] gets the rows
  #device_x.ravel() flattens x to: [x_0, ...]
  #using .sum() will add the results for the row
  rowSum = gpuarray.sum(device_W[i, :] * device_x.ravel())
  
  #Strores the sum into the first row of the result vector
  device_result[0, i] = rowSum

# Add the bias to the result vector
gpu_f = device_result + device_b

# transfers the final result from GPU to CPU
#if = gpu_f.get()

# ---------------- OUTPUT Result ----------------
print("\nResult: ")
print(gpu_f)


# Verify with NumPy computation on CPU
# np.dot(host_W, host_x).T transposes to (1,3)
cpu_result = np.dot(host_W, host_x).T + host_b

#checking result on CPU 
print("\nVerification (CPU computation):")
print(cpu_result)

#if it matches in both the CPU and GPU
print("\nMatch:", np.allclose(gpu_f.get(), cpu_result))













