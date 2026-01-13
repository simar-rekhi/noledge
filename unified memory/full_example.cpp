#include <cuda_runtime_api.h>
#include <cstdlib>
#include <memory.h>
#include <stdio.h>
#include <cuda/cmath>
#include <ctime>
using namespace std

// kernel for vector addition
__global__ void vectorAddition(float* A, float* B, float* C, int vectorLength){
  //conversion of a multi-dimensional coordinate to a flattened coordinate
  int flat_coordinate = threadIdx.x + (blockIdx.x * blockDim.x);
  if (flat_coordinate < vectorLength){
    // perform addition
    C[flat_coordinate] = A[flat_coordinate] + B[flat_coordinate];
  }
}

// initialize array function
void initArray(float* A, int length){
  //seed with time(NULL)
  srand(time(NULL));
  for(int i = 0; i < length; i++){
    A[i] = rand() / RAND_MAX(float);
  }
}

// function for serial vector addition (for CPU compute)
void serialVectorAdd(float* A, float* B, float* C, int length){
  for(int i = 0; i < length; i++){
    C[i] = A[i] + B[i];
  }
}

// vector approximation function
bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}



// unified memory usage
void unified_memory_example(int vectorLength){
  // pointer to vectors
  float* A = nullptr;
  float* B = nullptr;
  float* C = nullptr;
  float* comparison_matrix = (float*)malloc(vectorLength*sizeof(float));

  // use unified memory to allocate buffers
  cudaMallocManaged(&A, vectorLength*sizeof(float));
  cudaMallocManaged(&B, vectorLength*sizeof(float));
  cudaMallocManaged(&C, vectorLength*sizeof(float));

  //initialize vectors on host via initArray()
  initArray(A, vectorLength);
  initArray(B, vectorLength);

  // launching kernel
  // mention intrinsics
  int threads = 256;
  int blocks = cuda::ceil_div(vector_length, threads);

  // using triple chevron notation to mention intrinsics
  vectorAddition<<<blocks, threads>>>(A, B, C, vectorLength);

  cudaDeviceSynchronize();

  // perform serial compute via serialVectorAdd()
  serialVectorAdd(A, B, comparison_matrix, vectorLength);

  // confirm correctness
  if vectorApproximatelyEqual(C, comparison_matrix, vectorLength){
    printf("unfied memory successful\n");
  }
  else{
    printf("unified memory failes\n");

  // clean uo
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(comparison_matrix);
}



int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = atoi(argv[1]);
    }
    unified_memory_example(vectorLength);		
    return 0;
}
