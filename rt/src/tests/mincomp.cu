/*
   mincomp.h

   Test code. Part of the GGC source code. 

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu> 
*/

#include "gg.h"
#include "ggcuda.h"

__global__ 
void test_kernel(int nt, int *compweights, int *comptids, int *components, int *weights, LockArray l)  { 
  int tid = TID_1D;
  bool retry = true;
  
  if(tid < nt) {
    int compid = components[tid];
    while(retry) {
      if(l.acquire_or_fail(compid)) {
	if(compweights[compid] == 0 || (compweights[compid] > weights[tid])) {
	  compweights[compid] = weights[tid];
	  comptids[compid] = tid;
	}
	l.release(compid);
	retry = false;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  
  if(argc == 1) {
    fprintf(stderr, "Usage: %s comp_weight.dat\n", argv[0]);
    exit(0);
  }

  FILE *f = fopen(argv[1], "r");
  int nthreads;

  if(!f) {
    fprintf(stderr, "Failed to open file %s\n", argv[1]);
    exit(1);
  }

  if(fscanf(f, "%d", &nthreads) > 0) {
    fprintf(stderr, "reading %d lines\n", nthreads);

    Shared<int> comps(nthreads), weights(nthreads);
    int *ccomp = comps.cpu_wr_ptr(), *cweight = weights.cpu_wr_ptr();
    int maxcomp = 0;

    for(int i = 0; i < nthreads; i++) {
      if(fscanf(f, "%d %d",  &ccomp[i], &cweight[i]) != 2) {
	fprintf(stderr, "Failed to read on line %d\n", i);
	exit(1);
      }
      assert(cweight[i] != 0);
      maxcomp = ccomp[i] > maxcomp ? ccomp[i] : maxcomp;
    }
    maxcomp += 1;
    fprintf(stderr, "max components: %d\n", maxcomp);

    LockArray cl(maxcomp);
    Shared<int> cw(maxcomp), ct(maxcomp);

    cw.cpu_wr_ptr(); ct.cpu_wr_ptr(); // initialize to 0
 
    int nblocks = ((nthreads + 255) / 256) * 256;

    test_kernel<<<nblocks, 256>>>(nthreads, cw.gpu_wr_ptr(), ct.gpu_wr_ptr(), comps.gpu_rd_ptr(), weights.gpu_rd_ptr(), cl);
    cudaDeviceSynchronize();

    int *ccw = cw.cpu_rd_ptr(), *cct = ct.cpu_rd_ptr();

    ccomp = comps.cpu_rd_ptr(); cweight = weights.cpu_rd_ptr();

    for(int i = 0; i < maxcomp; i++) {
      printf("%d: %d %d\n", i, cct[i], ccw[i]);
      if(ccw[i] != 0)
	assert(ccw[i] == cweight[cct[i]] && ccomp[cct[i]] == i);
    }
   
  } else {
    fprintf(stderr, "Failed to read number of entries\n");
  }
  
}
