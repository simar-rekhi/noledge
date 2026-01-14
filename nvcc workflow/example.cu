// example.cu is an illustration the NVCC compilation workflow
#include <stdio.h>

// simple kernel code that prints a statement
__global__ void kernel(){
  printf("Hello from CUDA Kernel\n");
}

// kernel launching function that takes care of declaring intrinsics
void kernel_launch(){
  int blocks = 1;
  int threads = 1;

  kernel<<<blocks, threads>>>();
  cudaDeviceSynchronize();
}

// main function that calls the func
int main(){
  kernel_launch();
  return 0;
}  
