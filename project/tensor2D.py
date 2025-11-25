import numpy as np
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
#column
class tensor2D:
    def __init__(self, kernel_tensor, q=2):
        #like the this in java (this.name)
        self.kernel_tensor = kernel_tensor
        #self.W = W.astype(np.float32)
        #self.X = X.astype(np.float32)
        #self.b = b.astype(np.float32)
        self.q = q
    # ----------------- Initilize CUDA ----------------- #
        driver.init()

        # Determines the number of GPUs
        self.num_gpus = driver.Device.count()


    def split_blocks(self, matrix, q):

        M, N = matrix.shape
        rows = M // q
        cols = N // q

        blocks = []
        for i in range(q):
            r_blocks = []

            for j in range(q):
                r_start = i * rows
                if(i == q - 1):
                   r_end = M
                else:
                   r_end = (i+1)*rows

                c_start = j * cols
                if(j == q - 1):
                   c_end = N
                else:
                   c_end = (j+1) * cols

                block = matrix[r_start:r_end, c_start:c_end]
                r_blocks.append(block)

            blocks.append(r_blocks)

        return blocks

    def partial_product(self, W, X):

        M, K1 = W.shape
        K2, N = X.shape
        q =self.q

        W_blocks = self.split_blocks(W, q)
        X_blocks = self.split_blocks(X, q)

        partials = [[None for _ in range(q)] for _ in range(q)]


        for i in range(q):
            for j in range(q):
                 M_block, _ = W_blocks[i][0].shape
                 _, N_block = X_blocks[0][j].shape
                 results = np.zeros((M_block, N_block), dtype=np.float32)
                 for k in range(q):
                    gpu_id = ((i*q+j)*q+k)%self.num_gpus
                    cxt = driver.Device(gpu_id).make_context()
                    try:
                        W_block = W_blocks[i][k].astype(np.float32)
                        X_block = X_blocks[k][j].astype(np.float32)

                        M_block, K_block = W_block.shape
                        K_block, N_block = X_block.shape

                        device_W = gpuarray.to_gpu(W_block.ravel())
                        device_X = gpuarray.to_gpu(X_block.ravel())
                        device_y = gpuarray.empty((M_block * N_block), dtype=np.float32)

                        mod = SourceModule(self.kernel_tensor)
                        kernel = mod.get_function("mat_mat")

                        BLOCK_SIZE = 32
                        block = (BLOCK_SIZE,BLOCK_SIZE,1)
                        grid = ((N_block + BLOCK_SIZE -1)//BLOCK_SIZE,
                                (M_block + BLOCK_SIZE-1)//BLOCK_SIZE)

                        kernel(device_W, device_X, device_y,
                                np.int32(M_block), np.int32(N_block), np.int32(K_block),
                                block = block,
                                grid = grid
                        )

                        cxt.synchronize()
                        partial = device_y.get().reshape(M_block, N_block)

                        # results
                        if results is None:
                            results = partial
                        else:
                            results += partial


                    finally:
                         # Clean up GPU context
                          cxt.pop()
                          cxt.detach()

                 partials[i][j] = results
        return partials

    def sum_partials(self, partials):
        q = self.q
        r_size = [partials[i][0].shape[0] for i in range(q)]
        c_size = [partials[i][j].shape[1] for i in range(q)]

        M = sum(r_size)
        N = sum(c_size)

        y = np.zeros((M, N), dtype=np.float32)

        r_start = 0
        for i in range(q):
            r_end = r_start + r_size[i]
            c_start = 0
            for j in range(q):
                #r_start = i * b_rows
                c_end = c_start + c_size

                y[r_start:r_end[i], c_start:c_end] = partials[i][j]
                c_start += c_size
            r_start += r_size
        return y


    def run(self, W, X, b=None):
        W = W.astype(np.float32)
        X = X.astype(np.float32)
        b = b.astype(np.float32)

        partials = self.partial_product(W,X)
        y_result = self.sum_partials(partials)
        y_result += b
        return y_result
