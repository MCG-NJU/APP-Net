#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  Function: nearest devoxelization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r   : resolution
    r2  : r**2
    r3  : r**3
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, 3, n]
    feat: voxel features, FloatTensor[b, c, s]
    out : point features, FloatTensor[b, c, n]
*/
__global__ void nearest_devoxelize_kernel(int b, int c, int n, int r, int r2, int r3,
                                    const int *__restrict__ coords,
                                    const float *__restrict__ feat,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  feat += batch_index * c * r3;
  out += batch_index * c * n;
  for (int i = index; i < n; i += stride) {
    int pos = coords[i] * r2 + coords[i + n] * r + coords[i + n + n];
    for(int j = 0;j<c;j++){
        //printf("f=%f,", feat[j*r3+pos]);
        atomicAdd(out+j*n+i, feat[j*r3+pos]);
    }
  }
}


/*
  Function: nearest devoxelization (backward)
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    r3     : voxel cube size = voxel resolution ** 3
    ind    : voxel index of each point, IntTensor[b, n]
    grad_y : grad outputs, FloatTensor[b, c, n]
    grad_x : grad inputs, FloatTensor[b, c, s]
*/
__global__ void nearest_devoxelize_grad_kernel(int b, int c, int n, int r, int r2, int r3,
                                         const int *__restrict__ coords,
                                         const float *__restrict__ grad_y,
                                         float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  grad_x += batch_index * c * r3;
  grad_y += batch_index * c * n;
  for (int i = index; i < n; i += stride) {
    int pos = coords[i] * r2 + coords[i + n] * r + coords[i + n + n];
    for (int j = 0; j < c; j++) {
        atomicAdd(grad_x + j * r3 + pos, grad_y[j * n + i]);
    }
  }
}

void nearest_devoxelize_kernel_wrapper(int b, int c, int n, int r, int r2, int r3,
                          const int *inds, const float *feat,
                          float *outs) {
  nearest_devoxelize_kernel<<<b, optimal_num_threads(n)>>>(
      b, c, n, r, r2, r3, inds, feat, outs);
  CUDA_CHECK_ERRORS();
}

void nearest_devoxelize_grad_kernel_wrapper(int b, int c, int n, int r, int r2, int r3, const int *inds,
                               const float *grad_y,
                               float *grad_x) {
  nearest_devoxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(
      b, c, n, r, r2, r3, inds, grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}
