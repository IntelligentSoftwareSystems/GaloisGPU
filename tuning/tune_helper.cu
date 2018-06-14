/* -*- mode: c++ -*- */

#include <cuda.h>
#include <stdio.h>

void dump_settings_C(cudaDeviceProp &cu, int &rt_version, int &drv_version) {
  printf("#define GPU_NAME \"%s\"\n", cu.name);
  printf("#define GPU_VERSION_MAJOR %d\n#define GPU_VERSION_MINOR %d\n", cu.major, cu.minor);
  printf("#define RT_VERSION %d\n", rt_version);
  printf("#define DRV_VERSION %d\n", drv_version);
}

void dump_settings_python(cudaDeviceProp &cu, int &rt_version, int &drv_version) {
  printf("GPU_NAME = \"%s\"\n", cu.name);
  printf("GPU_VERSION_MAJOR = %d\nGPU_VERSION_MINOR = %d\n", cu.major, cu.minor);
  printf("RT_VERSION = %d\n", rt_version);
  printf("DRV_VERSION = %d\n", drv_version);

  /* could be used */
  printf("MAX_TPB = %d\n", cu.maxThreadsPerBlock);
  printf("MAX_THREADS_PER_SM = %d\n", cu.maxThreadsPerMultiProcessor);
  
  /* needed by ptatuner */
  if(cu.major <= 2)
    printf("MAX_TB_PER_SM = %d\n", 8);
  else if(cu.major >= 3) 
    printf("MAX_TB_PER_SM = %d\n", 16); /* conservative */
}

int main(int argc, char *argv[]) {
  cudaError_t err;
  cudaDeviceProp cu;
  int rt_version, drv_version;
  
  /* if multiple cards, use CUDA environment variable CUDA_VISIBLE_DEVICES to control card which is zero */
  /* http://docs.nvidia.com/cuda/cuda-c-programming-guide/#env-vars */

  if((err = cudaGetDeviceProperties(&cu, 0)) == cudaSuccess) {
    if((err = cudaDriverGetVersion(&drv_version)) != cudaSuccess) {
      fprintf(stderr, "Unable to get driver version, error: %d (%s)\n", err, cudaGetErrorString(err));
      exit(1);
    }

    if((err = cudaRuntimeGetVersion(&rt_version)) != cudaSuccess) {
      fprintf(stderr, "Unable to get runtime version, error: %d (%s)\n", err, cudaGetErrorString(err));
      exit(1);
    }
  } else {
    fprintf(stderr, "Unable to get CUDA device properties, error: %d (%s)\n", err, cudaGetErrorString(err));
    exit(1);
  }

  if(argc == 1 || (argc == 2 && strcmp(argv[1], "c") == 0))
    dump_settings_C(cu, rt_version, drv_version);
  else if (argc == 2 && strcmp(argv[1], "python") == 0)
    dump_settings_python(cu, rt_version, drv_version);
}
