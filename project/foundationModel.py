import numpy as np
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from tensor import tensor1D
from simpleTensor import SimpleTensor
from tensor_2D import tensor2D
import time

driver.init()

# Kenerel for the matrix-vector multiplication
kernel_tensor = """
  //activation

#include <math.h>
__device__ float gelu(float x){
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    float x3 = x * x * x;
    float inner = k0 * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));


}
__global__ void matrix_vec(float *W, float *x, float *b, float *y, int WIDTH){
    int row = blockIdx.x;
    if(row>= gridDim.x) return;

    float sum = 0.0f;
    for(int col = 0; col < WIDTH; col++){

       sum += W[row * WIDTH + col] * x[col];
    }
    //multiply corresponding elements and add the bias
    y[row] = gelu(sum + b[row]);
}

__global__ void mat_mat(float *W, float *X, float *y, int M, int N, int K){

   int col = blockIdx.x * blockDim.x + threadIdx.x ;
   int row = blockIdx.y * blockDim.y + threadIdx.y ;

   float sum = 0.0f;
   if (row < M && col < N) {
      for (int k = 0 ; k < K ; k++) {
         sum += W[row * K + k ] * X[k * N + col] ;
      }

      y[row * N + col] = sum;
   }
}
"""

# ----------------- FROM LAB 5 -----------------
WIDTH = 14

q = 2 # qxq block

#Random square matrix
host_W = np.random.randint(1,10, size=(WIDTH, WIDTH)).astype(np.float32)

#Random column vector (x)
host_x = np.random.randint(1,10, size=(WIDTH,)).astype(np.float32)

#Random bias vecotr (b)
host_b = np.random.randint(1,10,size=(WIDTH,)).astype(np.float32)

#m generate 2nd matrix
host_X = np.random.randint(1,10, size=(WIDTH, WIDTH)).astype(np.float32)

num_gpus = driver.Device.count()
print("\nNumber of GPUs detected:", num_gpus)

print("\nInput matrix W:")
print(host_W)
print("\nInput vector x:")
print(host_x)
print("\nBias vector b:")
print(host_b)


# ------------------- Simple Tensor ------------------- #
start = time.time()

simple = SimpleTensor(kernel_tensor, WIDTH)
simple_results = simple.run(host_W, host_x, host_b)

end = time.time()
# ------------------- 1D Tensor ------------------- #
start = time.time()

tensor = tensor1D(kernel_tensor, WIDTH)
output_results = tensor.run(host_W, host_x, host_b)

end = time.time()

# ------------------- 2D Tensor ------------------- #
start = time.time()
tensor_2d = tensor2D(kernel_tensor, q)
results_2D = tensor_2d.run(host_W, host_X, host_b)

end = time.time()

# -------------------- OUTPUTS -------------------- #
print("\nSimple Tensor Parallelism:")
print(simple_results)
print("Simple Tensor run time:", end - start, "seconds")

print("\nResult for 1D Tensor Parallelism:")
print(output_results)
print("1D Tensor run time:", end - start, "seconds")

print("\nResult for 2D Tensor Parallelism:")
print(results_2D)
print("2D Tensor run time:", end - start, "seconds")