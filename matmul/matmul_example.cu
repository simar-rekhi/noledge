// matmul_example.cu is a detailed yet simple approach to matrix multiplication on CUDA in C++

// naive approach for sequential calculation on host i.e., CPU only
void matmul_cpu(float* A, float* B, float* C, int M, int N, int K){
  // A of size M*K and B of size K*N. So, C of size M*N
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      float sum = 0.0f;
      for(int k = 0; k < K; k++){
        sum += A[i*K + k] * B[k*N + j];
      }
      C[i*N + j] = sum;
    }
  }
}

// naive CUDA kernel approach
__global__ void matmul_gpu_naive(float* A, float* B, float* C, int M, int N, int K){
  // each thread shall compute one element of C

  // flatten the coordinate
  int row = threadIdx.y + blockIdx.y * blockDim.y;  // rows stacked vertically, so y
  int col = threadIdx.x + blockIdx.x * blockDim.x;  // cols stacked horizontally, so x

  if (row >= M || col >= N){
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < K; k++){
    sum += A[row*K + k] * B[k*N + col];
  }

  C[row*N + col] = sum;
}

// optimized CUDA kernel approach with shared memory tiling
#define TILE 16
__global__ void matmul_gpu_optimized(float* A, float* B, float* C, int M, int N, int K){
  __shared__ float A_shared[TILE][TILE];
  __shared__ float B_shared[TILE][TILE];

  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  float sum = 0.0f;
  int num_tiles = (K + TILE - 1) / TILE;

  for (int t = 0; t < num_tiles; t++){
    int tiledRow = row;
    int tiledColA = t * TILE + threadIdx.x;

    int tiledRowB = t* TILE + threadIdx.y;
    int tiledCol = col;

    if (tiledRow < M && tiledColA < K)
      A_shared[threadIdx.y][threadIdx.x] = A[tiledRow*K + tiledColA];
    else
      A_shared[threadIdx.y][threadIdx.x] = 0.0f;

    if (tiledRowB < K && tiledCol < N)
      B_shared[threadIdx.y][threadIdx.x] = B[tiledRowB*N + tiledCol];
    else 
      B_shared[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for(int k = 0; k < TILE; k++){
      sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N)
        C[row*N + col] = sum;
}
