/** Survey propagation -*- CUDA -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Survey Propagation
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 */
#include "lonestargpu.h"

#define WORKPERTHREAD	1
#define FIXEDFACTOR	4
#define EPSILON		0.001
#define MAXITERATION	100

#define AllocateCuda(var, howmany, type)	{ if (cudaMalloc((void **)&var, (howmany) * sizeof(type)) != cudaSuccess) CudaTest("allocating var failed"); }
#define AllocateC(var, howmany, type)		{ var = (type *)malloc((howmany) * sizeof(type)); }

unsigned NCLAUSES, NLITERALS, LITPERCLA;
static unsigned *lperc;
bool inclause(unsigned newlit, unsigned iilperc) {
	for (unsigned ii = 0; ii < iilperc; ++ii) {
		if (lperc[ii] == newlit) {
			return true;
		}
	}
	return false;
}
unsigned getnextliteral() {
	static unsigned iilperc = 0;

	if (iilperc == LITPERCLA) {
		iilperc = 0;
	}
	unsigned newlit;
	do {
		newlit = rand() % NLITERALS;
	} while (inclause(newlit, iilperc));
	++iilperc;
	return newlit;
}
void init(unsigned *l2c, unsigned *c2l, unsigned *cperl, bool *eisneg, float *eeta, bool *lsolved, bool *lvalue, float *lbias/*, float *leta*/) {
	unsigned mm, nn, kk;
	unsigned *noutgoing = (unsigned *)malloc(NLITERALS * sizeof(unsigned));
	for (nn = 0; nn < NLITERALS; ++nn) {
		noutgoing[nn] = 0;
		lsolved[nn] = false;				// init.
		lvalue[nn] = false;				// init.
		lbias[nn] = (float)(rand()) / (float)RAND_MAX;		//1.0;				// init.
		//leta[nn] = 1.0;					// init.
	}
	lperc = (unsigned *)malloc(LITPERCLA * sizeof(unsigned));
	for (mm = 0; mm < NCLAUSES; ++mm) {
		unsigned row = mm * LITPERCLA;
		for (kk = 0; kk < LITPERCLA; ++kk) {
			unsigned newlit = getnextliteral();
			c2l[row + kk] = newlit;
			++noutgoing[newlit];
			eisneg[row + kk] = (bool)(rand() % 2);		// init.
			eeta[row + kk] = (float)(rand()) / (float)RAND_MAX;	// init.
		}
	}
	free(lperc);
	// generated all clauses; copy information to literals.

	unsigned currsum = 0;
	for (nn = 0; nn < NLITERALS; ++nn) {
		l2c[nn] = currsum;
		currsum += noutgoing[nn];
		noutgoing[nn] = 0;	// for the next phase.
	}
	l2c[NLITERALS] = currsum;	// last extra entry.
	// populated l2c: found starting indices in cperl.

	for (mm = 0; mm < NCLAUSES; ++mm) {
		unsigned row = mm * LITPERCLA;
		for (kk = 0; kk < LITPERCLA; ++kk) {
			unsigned lit = c2l[row + kk];
			unsigned index = l2c[lit] + noutgoing[lit]++;
			cperl[index] = mm;
		}
	}
	// populated cperl.
	free(noutgoing);
}

__global__
void dinit(unsigned NLITERALS, float *lprod0, float *lprodP, float *lprodN) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < NLITERALS) {
		lprod0[id] = lprodP[id] = lprodN[id] = 0.5;
	}
}
__global__
void dupdatebias(unsigned nblocks, unsigned NCLAUSES, unsigned NLITERALS, unsigned LITPERCLA, unsigned nedges, unsigned *l2c, unsigned *c2l, unsigned *cperl, float *eeta, bool *eisneg, /*float *leta,*/ float *lbias, bool *lvalue, bool *lsolved, float *lprod0, float *lprodP, float *lprodN, unsigned *changed, float *gMaxBias, float *gAvgBiasSum, unsigned *gAvgBiasNum) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned wpt = (NLITERALS + nblocks * blockDim.x - 1) / nblocks / blockDim.x;	// WORKPERTHREAD;
	unsigned start = id * wpt;
	unsigned ii;
	float lmax = 0.0, lsum = 0.0; 
	unsigned lnum = 0;

	if (start < NLITERALS) {
	for (ii = 0; ii < wpt; ++ii) {
	    unsigned ll = start + ii;
	    if (lsolved[ll] == false) {
		// avoiding going over the edges of ll, assuming lprods will be updated by updateeta.
		float p0 = lprod0[ll];
		float pP = lprodP[ll];
		float pN = lprodN[ll];

		float ppos = (1.0 - pP) * pN;
		float pneg = (1.0 - pN) * pP;

		float biasP = ppos / (ppos + pneg + p0);
		float biasN = pneg / (ppos + pneg + p0);
		//    float bias0 = 1.0 - biasP - biasN;

		float d = biasP - biasN;
		if (d < 0.0) {
			d = biasN - biasP;
		}
		lbias[ll] = d;
		lvalue[ll] = (biasP > biasN);

		lmax = (lmax < d ? d : lmax);
		lsum += d;
		++lnum;

		if (d > EPSILON) {
			++*changed;
		}
	    }
	}
	*gMaxBias = (*gMaxBias < lmax ? lmax : *gMaxBias);		//atomicMax(gMaxBias, lmax);
	*gAvgBiasSum = (*gAvgBiasSum < lsum ? lsum : *gAvgBiasSum);	//atomicAdd(gAvgBiasSum, lsum);
	atomicAdd(gAvgBiasNum, lnum);
	}
}
#define MYLITPERCLA	3
__global__
void dupdateeta(unsigned nblocks, unsigned NCLAUSES, unsigned NLITERALS, unsigned LITPERCLA, unsigned nedges, unsigned *l2c, unsigned *c2l, unsigned *cperl, float *eeta, bool *eisneg, /*float *leta,*/ float *lbias, bool *lvalue, bool *lsolved, float *lprod0, float *lprodP, float *lprodN, unsigned *changed, unsigned *dummy) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned wpt = (NCLAUSES + nblocks * blockDim.x - 1) / nblocks / blockDim.x;	// WORKPERTHREAD;
	unsigned start = id * wpt;
	unsigned ii, kk;
	bool lchanged = false;

	if (start < NCLAUSES) {
	for (ii = 0; ii < wpt; ++ii) {
		unsigned cc = start + ii;
		unsigned row = cc * MYLITPERCLA;
		for (kk = 0; kk < MYLITPERCLA; ++kk) {
		    unsigned ll = c2l[row + kk];
		    if (lsolved[ll] == false) {
			float oldeta = eeta[row + kk];
			float neweta = 1.0;
			for (unsigned innerkk = 0; innerkk < MYLITPERCLA; ++innerkk) {
			  if (innerkk != kk) {
			  	  unsigned innerll = c2l[row + innerkk];
				  float inneroldeta = eeta[row + innerkk];
				  neweta *= lbias[innerll] / inneroldeta;
			  }
		        }
		  	//if (oldeta < neweta && neweta - oldeta > EPSILON || neweta < oldeta && oldeta - neweta > EPSILON) {
			if (fabs(oldeta - neweta) > EPSILON) {
					eeta[row + kk] = neweta;
					// also update literal's products.
					float factor = (1.0 - neweta);
					if (1.0 - oldeta > EPSILON) {
						factor /= (1.0 - oldeta);
					}
					lprod0[ll] *= factor;
					if (eisneg[row + kk]) {
						lprodN[ll] *= factor;
					} else {
						lprodP[ll] *= factor;
					}
					//++*changed;	// atomicInc(changed, nedges);
					lchanged = true;
				  }
		    	}
		}
	}
	}
	if (lchanged) {
		*changed = true;
	}
}
__global__
void ddecimate(unsigned nblocks, unsigned NCLAUSES, unsigned NLITERALS, unsigned LITPERCLA, unsigned nedges, unsigned *l2c, unsigned *c2l, unsigned *cperl, float *eeta, bool *eisneg, /*float *leta,*/ float *lbias, bool *lvalue, bool *lsolved, float *lprod0, float *lprodP, float *lprodN, unsigned *changed, float *gMaxBias, float *gAvgBiasSum, unsigned *gAvgBiasNum) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned wpt = (NLITERALS + nblocks * blockDim.x - 1) / nblocks / blockDim.x;	// WORKPERTHREAD;
	unsigned start = id * wpt;
	float limit = 0.0;
	unsigned ii;

	if (start < NLITERALS) {
	if (id == 0) {
		if (*gAvgBiasNum > 0) {
			float avgB = *gAvgBiasSum / *gAvgBiasNum;
			limit = *gMaxBias - 0.75 * avgB;
		} else {
			limit = *gMaxBias;
		}
	}
	__syncthreads();
	for (ii = 0; ii < wpt; ++ii) {
		if (lsolved[start + ii] == false && lbias[start + ii] >= limit) {
			lsolved[start + ii] = true;
			lvalue[start + ii] = true;
		}
	}
	}
}
__global__
void dverifysolution(unsigned nblocks, unsigned NCLAUSES, unsigned NLITERALS, unsigned LITPERCLA, unsigned nedges, unsigned *l2c, unsigned *c2l, unsigned *cperl, float *eeta, bool *eisneg, /*float *leta,*/ float *lbias, bool *lvalue, bool *lsolved, float *lprod0, float *lprodP, float *lprodN, unsigned *changed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned wpt = (NCLAUSES + nblocks * blockDim.x - 1) / nblocks / blockDim.x;	// WORKPERTHREAD;
	unsigned start = id * wpt;
	unsigned ii, kk;

	if (start < NCLAUSES) {
	unsigned nclausessatisfied = 0;
	for (ii = 0; ii < wpt; ++ii) {
		unsigned cc = start + ii;
		unsigned row = cc * LITPERCLA;
		for (kk = 0; kk < LITPERCLA; ++kk) {
			unsigned ll = c2l[row + kk];
			if (eisneg[row + kk] && lvalue[ll] == false || eisneg[row + kk] == false && lvalue[ll]) {
				++nclausessatisfied;
				break;
			}
		}
	}
	atomicAdd(changed, nclausessatisfied);
	}
}
__global__
void dverifyliterals(unsigned nblocks, unsigned NCLAUSES, unsigned NLITERALS, unsigned LITPERCLA, unsigned nedges, unsigned *l2c, unsigned *c2l, unsigned *cperl, float *eeta, bool *eisneg, /*float *leta,*/ float *lbias, bool *lvalue, bool *lsolved, float *lprod0, float *lprodP, float *lprodN, unsigned *changed, unsigned *dummy) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned wpt = (NLITERALS + nblocks * blockDim.x - 1) / nblocks / blockDim.x;	// WORKPERTHREAD;
	unsigned start = id * wpt;
	unsigned ii;

	if (start < NLITERALS) {
	unsigned nliteralssatisfied = 0;
	unsigned nliteralssolved = 0;
	for (ii = 0; ii < wpt; ++ii) {
		unsigned ll = start + ii;
		if (lvalue[ll]) {
			++nliteralssatisfied;
		}
		if (lsolved[ll]) {
			++nliteralssolved;
		}
	}
	atomicAdd(changed, nliteralssatisfied);
	atomicAdd(dummy, nliteralssatisfied);
	}
}
#define DISTANCETHRESHOLD	150
int main(int argc, char *argv[])

{
	unsigned int hnedges;
	unsigned *changed, hchanged;
	int iteration = 0;
	unsigned *l2c, *c2l, *cperl, *hl2c, *hc2l, *hcperl;
	float *eeta, /**leta,*/ *lbias, *heeta, /**hleta,*/ *hlbias;
	bool *eisneg, *lvalue, *lsolved, *heisneg, *hlvalue, *hlsolved;
	float *lprod0, *lprodP, *lprodN;
	float floatzero = 0.0;
	unsigned intzero = 0;
	float *gMaxBias, *gAvgBiasSum;
	unsigned *gAvgBiasNum;
	unsigned *dummy, hdummy;
	float fdummy;
	KernelConfig kconf;

	float starttime, endtime;
	int runtime;


	cudaFuncSetCacheConfig(dupdateeta, cudaFuncCachePreferL1);

	if (argc != 5) {
		printf("Usage: %s seed M N K\n", argv[0]);
		exit(1);
	}

	cudaGetLastError();

	unsigned argno = 0;
	srand(atoi(argv[++argno]));		// seed.
	NCLAUSES = atoi(argv[++argno]);		// M.
	NLITERALS = atoi(argv[++argno]);	// N.
	LITPERCLA = atoi(argv[++argno]);	// K.
	hnedges = NCLAUSES * LITPERCLA;


	hl2c = (unsigned *)malloc((NLITERALS + 1) * sizeof(unsigned));
	hc2l = (unsigned *)malloc(hnedges * sizeof(unsigned));
	hcperl = (unsigned *)malloc(hnedges * sizeof(unsigned));

	heeta = (float *)malloc(hnedges * sizeof(float));
	heisneg = (bool *)malloc(hnedges * sizeof(bool));
	//hleta = (float *)malloc(NLITERALS * sizeof(float));
	hlbias = (float *)malloc(NLITERALS * sizeof(float));
	hlvalue = (bool *)malloc(NLITERALS * sizeof(bool));
	hlsolved = (bool *)malloc(NLITERALS * sizeof(bool));

	printf("populating data structures: M=%d, N=%d, K=%d.\n", NCLAUSES, NLITERALS, LITPERCLA);
	init(hl2c, hc2l, hcperl, heisneg, heeta, hlsolved, hlvalue, hlbias/*, hleta*/);

	if (cudaMalloc((void **)&l2c, (NLITERALS + 1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating l2c failed");
	if (cudaMalloc((void **)&c2l, hnedges * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating c2l failed");
	if (cudaMalloc((void **)&cperl, hnedges * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating cperl failed");

	if (cudaMalloc((void **)&eeta, hnedges * sizeof(float)) != cudaSuccess) CudaTest("allocating eeta failed");
	if (cudaMalloc((void **)&eisneg, hnedges * sizeof(bool)) != cudaSuccess) CudaTest("allocating eisneg failed");
	//if (cudaMalloc((void **)&leta, NLITERALS * sizeof(float)) != cudaSuccess) CudaTest("allocating leta failed");
	if (cudaMalloc((void **)&lbias, NLITERALS * sizeof(float)) != cudaSuccess) CudaTest("allocating lbias failed");
	if (cudaMalloc((void **)&lvalue, NLITERALS * sizeof(bool)) != cudaSuccess) CudaTest("allocating lvalue failed");
	if (cudaMalloc((void **)&lsolved, NLITERALS * sizeof(bool)) != cudaSuccess) CudaTest("allocating lsolved failed");
	if (cudaMalloc((void **)&lprod0, NLITERALS * sizeof(float)) != cudaSuccess) CudaTest("allocating lprod0 failed");
	if (cudaMalloc((void **)&lprodP, NLITERALS * sizeof(float)) != cudaSuccess) CudaTest("allocating lprodP failed");
	if (cudaMalloc((void **)&lprodN, NLITERALS * sizeof(float)) != cudaSuccess) CudaTest("allocating lprodN failed");


	cudaMemcpy(l2c, hl2c, (NLITERALS + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(c2l, hc2l, hnedges * sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(cperl, hcperl, hnedges * sizeof(unsigned), cudaMemcpyHostToDevice);

	cudaMemcpy(eeta, heeta, hnedges * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(eisneg, heisneg, hnedges * sizeof(bool), cudaMemcpyHostToDevice);
	//cudaMemcpy(leta, hleta, NLITERALS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(lbias, hlbias, NLITERALS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(lvalue, hlvalue, NLITERALS * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(lsolved, hlsolved, NLITERALS * sizeof(bool), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating changed failed");
	if (cudaMalloc((void **)&gMaxBias, sizeof(float)) != cudaSuccess) CudaTest("allocating gMaxBias failed");
	if (cudaMalloc((void **)&gAvgBiasSum, sizeof(float)) != cudaSuccess) CudaTest("allocating gAvgBiasSum failed");
	if (cudaMalloc((void **)&gAvgBiasNum, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating gAvgBiasNum failed");
	if (cudaMalloc((void **)&dummy, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating dummy failed");

	kconf.setProblemSize(NLITERALS);
	dinit<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (NLITERALS, lprod0, lprodP, lprodN);
	CudaTest("dinit failed");

	kconf.setProblemSize(NCLAUSES);
	unsigned outeriteration = 0;

	printf("solving.\n");
	starttime = rtclock();

	do {
		do {
			++iteration;
			//printf("iteration %d, nblocks=%d, wpt=%d.\n", iteration, NBLOCKS*CFACTOR, (NCLAUSES + NBLOCKS * CFACTOR * BLOCKSIZE - 1)/NBLOCKS/CFACTOR/BLOCKSIZE);
			hchanged = 0;
			cudaMemcpy(changed, &hchanged, sizeof(unsigned), cudaMemcpyHostToDevice);
			cudaMemcpy(dummy, &hchanged, sizeof(unsigned), cudaMemcpyHostToDevice);
			dupdateeta <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (kconf.getNumberOfBlocks(), NCLAUSES, NLITERALS, LITPERCLA, hnedges, l2c, c2l, cperl, eeta, eisneg, /*leta,*/ lbias, lvalue, lsolved, lprod0, lprodP, lprodN, changed, dummy);
			CudaTest("updateeta failed");
			cudaMemcpy(&hchanged, changed, sizeof(unsigned), cudaMemcpyDeviceToHost);
			cudaMemcpy(&hdummy, dummy, sizeof(unsigned), cudaMemcpyDeviceToHost);
			//printf("changed eta of %d edges, iteration=%d.\n", hchanged, iteration);
		} while (hchanged && iteration < MAXITERATION);

		cudaMemcpy(gMaxBias, &floatzero, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gAvgBiasSum, &floatzero, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gAvgBiasNum, &intzero, sizeof(unsigned), cudaMemcpyHostToDevice);

		dupdatebias <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (kconf.getNumberOfBlocks(), NCLAUSES, NLITERALS, LITPERCLA, hnedges, l2c, c2l, cperl, eeta, eisneg, /*leta,*/ lbias, lvalue, lsolved, lprod0, lprodP, lprodN, changed, gMaxBias, gAvgBiasSum, gAvgBiasNum);
		CudaTest("updatebias failed");
		cudaMemcpy(&hchanged, changed, sizeof(unsigned), cudaMemcpyDeviceToHost);
		cudaMemcpy(&fdummy, gMaxBias, sizeof(float), cudaMemcpyDeviceToHost);
		//printf("updated bias of %d literals, maxbias=%.2f, ", hchanged, fdummy);
		cudaMemcpy(&fdummy, gAvgBiasSum, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&hdummy, gAvgBiasNum, sizeof(unsigned), cudaMemcpyDeviceToHost);
		//printf(" avgbias=%.2f.\n", fdummy/hdummy);

		if (hchanged > 0) {
			kconf.setProblemSize(NLITERALS);
			ddecimate <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (kconf.getNumberOfBlocks(), NCLAUSES, NLITERALS, LITPERCLA, hnedges, l2c, c2l, cperl, eeta, eisneg, /*leta,*/ lbias, lvalue, lsolved, lprod0, lprodP, lprodN, changed, gMaxBias, gAvgBiasSum, gAvgBiasNum);
			CudaTest("decimate failed");
			//printf("decimate: fixed %d literals, iteration=%d.\n", hchanged, iteration);
			kconf.setProblemSize(NCLAUSES);
		}
		++outeriteration;
	} while (hchanged > 0 && outeriteration < MAXITERATION);
	endtime = rtclock();
	runtime = (int) (1000.0f * (endtime - starttime));
	printf("%d ms.\n", runtime);
	
	// cleanup left to the OS.

	return 0;
}
