/** Breadth-first search -*- CUDA -*-
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
 * Example breadth-first search application for demoing Galois system.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 */

#include "lonestargpu.h"

__global__
void initialize(foru *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}


__global__
void dverifysolution(foru *dist, Graph graph, unsigned *nerr) {
	unsigned int nn = (blockIdx.x * blockDim.x + threadIdx.x);
	  if (nn < graph.nnodes) {
		unsigned int nsrcedges = graph.getOutDegree(nn);
		for (unsigned ii = 0; ii < nsrcedges; ++ii) {
			unsigned int u = nn;
			unsigned int v = graph.getDestination(u, ii);
			foru wt = 1;
			if (wt > 0 && dist[u] + wt < dist[v]) {
				++*nerr;
			}
		}
	  }	
}

__device__
bool processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	dst = graph.getDestination(src, ii);
	if (dst >= graph.nnodes) return 0;

	foru wt = 1;

	foru altdist = dist[src] + wt;
	if (altdist < dist[dst]) {
	 	foru olddist = atomicMin(&dist[dst], altdist);
		if (altdist < olddist) {
			return true;
		} 
		// someone else updated distance to a lower value.
	}
	return false;
}
__device__
bool processnode(foru *dist, Graph &graph, unsigned work) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn = work;
	if (nn >= graph.nnodes) return 0;
	bool changed = false;
	
	unsigned neighborsize = graph.getOutDegree(nn);
	for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		foru olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
			changed = true;
		}
	}
	return changed;
}

__global__
void drelax(foru *dist, Graph graph, bool *changed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x);

	for (unsigned ii = start; ii < end; ++ii) {
		if (processnode(dist, graph, ii)) {
			*changed = true;
		}
	}
}

int main(int argc, char *argv[]) {
	foru *dist;
	foru foruzero = 0.0;
	unsigned intzero = 0;
	bool *changed, hchanged;
	int iteration = 0;
	Graph hgraph, graph;
	unsigned *nerr, hnerr;
	KernelConfig kconf;

	double starttime, endtime;

	cudaFuncSetCacheConfig(drelax, cudaFuncCachePreferShared);
	if (argc != 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}

	cudaGetLastError();

	hgraph.read(argv[1]);
	//hgraph.optimize();
	hgraph.cudaCopy(graph);

	kconf.setProblemSize(graph.nnodes);

	if (cudaMalloc((void **)&dist, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating dist failed");

	kconf.setMaxThreadsPerBlock();
	printf("initializing.\n");
	initialize <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph.nnodes);
	CudaTest("initializing failed");

	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");
	if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");

	printf("solving.\n");
	starttime = rtclock();

	do {
		++iteration;
		hchanged = false;

		cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice);

		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed);
		CudaTest("solving failed");

		cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost);
	} while (hchanged);
	endtime = rtclock();
	
	printf("\truntime = %.3lf ms.\n", 1000 * (endtime - starttime));

	cudaMemcpy(nerr, &intzero, sizeof(intzero), cudaMemcpyHostToDevice);
	kconf.setMaxThreadsPerBlock();
	printf("verifying.\n");
	dverifysolution<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, nerr);
	CudaTest("dverifysolution failed");
	cudaMemcpy(&hnerr, nerr, sizeof(hnerr), cudaMemcpyDeviceToHost);
	printf("\tno of errors = %d.\n", hnerr);
	

	// cleanup left to the OS.

	return 0;
}
