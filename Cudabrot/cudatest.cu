#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 100;
  float *x = new float;
  float *y = new float;

  // Allocate Unified Memory – accessible from CPU or GPU
  std::cout << "ok0 " << std::endl;;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  std::cout << "ok1 " << std::endl;;
  
  
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    std::cout << "x " << i << std::endl;;
    x[i] = 0.0;
    y[i] = 0.0;
  }
  std::cout << "ok2 " << std::endl;
  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);
  std::cout << "ok3 " << std::endl;
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  std::cout << "ok4 " << std::endl;
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
  std::cout << "ok5 " << std::endl;
  // Free memory
  cudaFree(x);
  cudaFree(y);
  std::cout << "ok6 " << std::endl;
  return 0;
}