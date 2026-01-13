int main(){

  void unified_memory_example(int vectorLength) {
    // create pointer to the vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    // use unified memory cudaMallocManaged to allocate buffers to each of the vectors
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // initialized the vectors on host ie CPU
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // launch the kernel & specify the intrinsics
    // here, unfied memory makes sure that A, B, C are accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);

    vectorAddition<<<blocks, threads>>>(A, B, C, vectorLength);

    // wait for the kernel to complete executio and then run synchronization
    cudaDeviceSynchronization();

    // let us perform the exact same task on the CPU as well
    serialVectorAddition(A, B, comparisonMatrix, vectorLength);

    // analyze results from both CPU and GPU
    if (vectorApproximatelyEqual(C, comparisonMatrix, vectorLength)){
      printf("Unified Memory: CPU & GPU answers match\n");
    }
    else{
      printf("Unified Memory Error\n");
    }

    // perform clean up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonMatrix);
  }
