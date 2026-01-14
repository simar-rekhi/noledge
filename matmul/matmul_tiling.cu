// this is a rough example of memory tiling for matmul on gpu

#define TILE_WIDTH 16
// this basically creates each tile of size 16 * 16 i.e., 256 threads per tile

__global__ void matmul(float* A, float* B, float* C, int M, int K, int N){
  // C is the resultant matrix
  // A is of size (M x K) and B is of size (K x N), so C is of size (M x N) 

  // find out number of tiles in C using ceil division
  int num_tiles = (TILE_WIDTH + K - 1) / TILE_WIDTH;

  // creating shared memory tiles for A and B
  __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

  // developing coordinate system that keeps track of A_tile, B_tile and C
  // to parse rows and columns these are the global coordinates
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  // loading tile A and then tile B
  for (int t = 0; t < num_tiles; t++){  
    // t is the tile index that goes from 0 to num_tiles
    global_row_A = row;
    global_col_A = t*TILE_WIDTH + threadIdx.x;

    if (global_row_A < M && global_col_A < K){
      A_tile[threadIdx.y][threadIdx.x] = A[global_row_A * K + global_col_A];
    else
      A_tile[threadIdx.y][threadIdx.x] = 0.0f;


  global_col_B = col;
  global_row_B = threadIdx.y + t*TILE_WIDTH;

  if (global_col_B < N && global_row_B < K){
    B_tile[threadIdx.y][threadIdx.x] = B[global_row_B*N + global_col_B];
  else
    B_tile[threadIdx.y][threadIdx.x] = 0.0f;

  // sync threads of these tiles
  __syncthreads();

  // multiply and accumulate these tiles
  for (int k = 0; k < TILE_WIDTH; k++){
    sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];

    __syncthreads();
  }
}

  
