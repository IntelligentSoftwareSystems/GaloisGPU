/*
   la.cu

   Tests for LockArray. Part of the GGC source code. 

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu> 
*/

#include "lockarray.h"

// simple test of acquire_or_fail
__global__ void test (LockArray l) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool ac = false;

  ac = l.acquire_or_fail(0);

  //printf("%d %d %d\n", tid, l.is_locked(tid), l.glocks[tid]);

  if(ac) 
    l.release(0);
  
  printf("%d %d %d\n", tid, l.is_locked(tid), l.glocks[tid]);   
}

// test of acquire, hangs
__global__ void test2 (LockArray l) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool ac = false;

  ac = l.acquire(0);

  printf("%d %d %d\n", tid, l.is_locked(tid), l.glocks[tid]);

  if(ac)
    l.release(0);
  
  printf("%d %d %d\n", tid, l.is_locked(tid), l.glocks[tid]);
}

// another instance of acquire which hangs
__global__ void test3 (LockArray l) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool ac = false;

  while(true) {
    if(l.acquire_or_fail(0)) {
      printf("%d acquired!\n", tid);
      printf("%d %d %d\n", tid, l.is_locked(0), l.glocks[0]);
      printf("%d released!\n", tid);
      l.release(0);
      break;
    }
  }
  __syncthreads();
  
  //printf("%d %d %d\n", tid, l.is_locked(tid), l.glocks[tid]);
}

// this works once break is removed, hmm ...
__global__ void test3_1 (LockArray l) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool ac = true;

  while(ac) {
    if(l.acquire_or_fail(0)) {
      printf("%d acquired!\n", tid);
      printf("%d %d %d\n", tid, l.is_locked(0), l.glocks[0]);
      printf("%d released!\n", tid);
      l.release(0);
      ac = false;
    }
  }
  __syncthreads();
  
  //printf("%d %d %d\n", tid, l.is_locked(tid), l.glocks[tid]);
}

// first attempt at 2-D thread block, strange assignment of threads to
// physical warps
__global__ void test4 (LockArray l) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool ac = false;
  
  assert(blockDim.y != 1);

  printf("%d %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z, get_warpid());
  // if(threadIdx.x == 0) {
  //   //l.acquire(0);
  //   printf("%d %d %d %d\n", tid, l.is_locked(tid), l.glocks[tid], get_warpid());
  //   //l.release(0);
  // }
}

// this works!
__global__ void test5 (LockArray l, int *x) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool ac = false;
  
  //printf("%d %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z, get_warpid());
  if((threadIdx.x % 32) == 0) {
    int lockid = 0;
    l.acquire(0);
    int oldx = *x;
    *x = oldx + 1;
    threadfence();
    printf("%llu %d %d %d %d %d %d\n", clock64(), tid, lockid, l.is_locked(lockid), l.glocks[lockid], get_warpid(), oldx);
    l.release(0);
  }
}

// Example 5 from Arun Ramamurthy's thesis, similar to test3 above
__global__ void test6 (LockArray l, int *x) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool ac = false;
  
  //printf("%d %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z, get_warpid());
  bool c = true;
  while(c) {
    int lockid = 0;
    if(l.acquire_or_fail(lockid)) {
	int oldx = *x;
	*x = oldx + 1;
	threadfence();
	printf("%llu %d %d %d %d %d %d\n", clock64(), tid, lockid, l.is_locked(lockid), l.glocks[lockid], get_warpid(), oldx);
	l.release(lockid);
	c = false;
    }
  } 
      
}

int main(void) {
  LockArray x(10);
  Shared<int> y(1);

  // y.cpu_wr_ptr();
  // test5<<<1, 320>>>(x, y.gpu_wr_ptr());
  // cudaDeviceSynchronize();
  // printf("Y: %d\n", *(y.cpu_rd_ptr()));

  y.cpu_wr_ptr();
  //test6<<<1, 10>>>(x, y.gpu_wr_ptr());
  test3_1<<<1, 10>>>(x);
  cudaDeviceSynchronize();
  printf("Y: %d\n", *(y.cpu_rd_ptr()));

  return 0;
}
