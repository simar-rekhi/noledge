//example.cu is an example to cudaMemcpy that copies an array to GPU and back
#include <stdio.h>
#include <cuda_runtime.h>

// kernel to add one to each current element in the matix
__global__ void addOne(int* matrix, int length){
  //flatten the coordinate
  int flat_coord = threadIdx.x + blockIdx.x * blockDim.x;

  // check bounds and perform the calculation
  if (flat_coord < length){
    matrix[flat_coord] += 1;
  }

int main(){
    int n = 5;
    int sz = n * sizeof(int);

    // step 1 - allocate memory to host
    int matrix[5] = {1, 2, 3, 4, 5};

    // step 2 - allocate memory to device
    int* matrixPtr;
    cudaMalloc((void*)&matrixPtr, sz);

    // step 3 - copy host -> device
    cudaMemcpy(matrixPtr, matrix, sz, cudaMemcpyHostToDevice);

    // step 4 - launch kernel
    addOne<<<1, 32>>>(matrixPtr, n);  

    // step 5 - copy back device -> host
    cudaMemcpy(matrix, matrixPtr, sz, cudaMemcpyDeviceToHost);

    // step 6 - free memory
    cudaFree(matrixPtr);

    return 0;
}


// important thing to take care --> why was only cudaMalloc used here and not cudaMallocManaged
