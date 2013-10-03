/** Minimum spanning tree -*- CUDA -*-
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
 * @Description
 * Computes minimum spanning tree of a graph using Boruvka's algorithm.
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 */

#include "lonestargpu.h"

__global__ void dinit(float *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
		eleminwts[id] = MYINFINITY;
		minwtcomponent[id] = MYINFINITY;	
		goaheadnodeofcomponent[id] = graph.nnodes;
		phores[id] = 0;
		partners[id] = id;
		processinnextiteration[id] = false;
	}
}
__global__ void dfindelemin(float *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
		// if I have a cross-component edge,
		// 	find my minimum wt cross-component edge,
		//	inform my boss about this edge e (atomicMin).
		unsigned src = id;
		unsigned srcboss = cs.find(src);
		unsigned dstboss = graph.nnodes;
		foru minwt = MYINFINITY;
		unsigned degree = graph.getOutDegree(src);
		for (unsigned ii = 0; ii < degree; ++ii) {
			foru wt = graph.getWeight(src, ii);
			if (wt < minwt) {
				unsigned dst = graph.getDestination(src, ii);
				unsigned tempdstboss = cs.find(dst);
				if (srcboss != tempdstboss) {	// cross-component edge.
					minwt = wt;
					dstboss = tempdstboss;
				}
			}
		}
		dprintf("\tminwt[%d] = %.f\n", id, minwt);
		eleminwts[id] = minwt;
		if (minwt < minwtcomponent[srcboss] && srcboss != dstboss) {
			// inform boss.
			foru oldminwt = atomicMin(&minwtcomponent[srcboss], minwt);
			if (oldminwt > minwt) {
				partners[id] = dstboss;
				goaheadnodeofcomponent[srcboss] = id;	// threads with same wt edge will race.
				dprintf("\tpartner[%d(%d)] = %d init, eleminwts[id]=%d\n", id, srcboss, dstboss, eleminwts[id]);
			}
		}
	}
}
__global__ void dfindcompmin(float *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
		unsigned srcboss = cs.find(id);
		unsigned dstboss = cs.find(partners[id]);
		if (id != partners[id] && srcboss != dstboss && eleminwts[id] != MYINFINITY && minwtcomponent[srcboss] == eleminwts[id] && dstboss != id && goaheadnodeofcomponent[srcboss] == id) {	// my edge is min outgoing-component edge.
			processinnextiteration[id] = true;
		}
	}
}
__global__ void dfindcompmintwo(float *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes && processinnextiteration[id]) {
			unsigned srcboss = cs.find(id);
			unsigned dstboss = cs.find(partners[id]);
			dprintf("trying unify id=%d (%d -> %d)\n", id, srcboss, dstboss);
			if (cs.unify(srcboss, dstboss)) {
				atomicAdd(mstwt, eleminwts[id]);
			}
			eleminwts[id] = MYINFINITY;	// mark end of processing to avoid getting repeated.
			dprintf("\tcomp[%d] = %d.\n", srcboss, cs.find(srcboss));
	}
}
__global__ void dfindcompminundirected(float *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *tryagain) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < graph.nnodes) {
		unsigned srcboss = cs.find(id);
		unsigned dstboss = cs.find(partners[id]);
		unsigned dstbosspartner = partners[dstboss];
		unsigned dstbosspartnerboss = cs.find(dstbosspartner);

		if (srcboss == dstboss) {
			return;
		} else if (srcboss == dstbosspartnerboss) {	// 2-cycle.
			if (srcboss < dstboss) {
				partners[id] = id;	// break cycle.
				*tryagain = 1;
				printf("\tpartner[%d] = %d self\n", id, dstboss);
			}
		} else {
			partners[id] = dstbosspartnerboss;
			*tryagain = 1;
			printf("\tpartner[%d] = %d\n", id, dstboss);
		}
	}
}
__global__ void dfindcompminlock(float *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *tryagain, int inputid = -1) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inputid != -1) id = inputid;

	if (id < graph.nnodes) {
		unsigned srcboss = cs.find(id);
		unsigned dstboss = cs.find(partners[id]);
		if (srcboss != dstboss && eleminwts[id] != MYINFINITY && minwtcomponent[srcboss] == eleminwts[id] && dstboss != id) {	// my edge is min outgoing-component edge.
			// multiple threads may come here (same minwt edge), choose one.
			unsigned srcsema, dstsema;
			unsigned smaller = srcboss, larger = dstboss;
			if (smaller > larger) { unsigned tmp = smaller; smaller = larger; larger = tmp; }

			srcsema = atomicInc(&phores[smaller], graph.nnodes + 1);
			while (srcsema != 0) {
				atomicDec(&phores[smaller], graph.nnodes + 1);
				srcsema = atomicInc(&phores[smaller], graph.nnodes + 1);
			}
			// grab the destination lock.
			dstsema = atomicInc(&phores[larger], graph.nnodes + 1);
			while (dstsema != 0) {
				atomicDec(&phores[larger], graph.nnodes + 1);
				dstsema = atomicInc(&phores[larger], graph.nnodes + 1);
			}
			//printf("%d: unifying %d with %d\n", id, srcboss, dstboss);
			cs.unify(srcboss, dstboss);
			atomicAdd(mstwt, eleminwts[id]);
			eleminwts[id] = MYINFINITY;	// mark end of processing to avoid getting repeated.

			atomicDec(&phores[dstboss], graph.nnodes + 1);	// unlock.
			atomicDec(&phores[srcboss], graph.nnodes + 1);	// unlock.
		}
	}
}

__global__ void dfindcompminatomic(float *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *tryagain, int inputid = -1) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inputid != -1) id = inputid;

	if (id < graph.nnodes) {
		unsigned srcboss = cs.find(id);
		unsigned dstboss = cs.find(partners[id]);
		if (srcboss != dstboss && eleminwts[id] != MYINFINITY && minwtcomponent[srcboss] == eleminwts[id] && dstboss != id) {	// my edge is min outgoing-component edge.
			// multiple threads may come here (same minwt edge), choose one.
			unsigned srcsema = atomicInc(&phores[srcboss], graph.nnodes + 1);
			if (srcsema == 0) {	// my edge is chosen, grabbed source lock.
				// grab the destination lock.
				unsigned dstsema = atomicInc(&phores[dstboss], graph.nnodes + 1);
				if (dstsema == 0) {	// now do the honours, grabbed destination lock.
					//printf("%d: unifying %d with %d\n", id, srcboss, dstboss);
					cs.unify(srcboss, dstboss);
					atomicAdd(mstwt, eleminwts[id]);
					eleminwts[id] = MYINFINITY;	// mark end of processing to avoid getting repeated.
				} else {
					*tryagain = true;
					//printf("\tthread %d aborts, did not get lock on dst %d.\n", id, dstboss);
				}
				atomicDec(&phores[dstboss], graph.nnodes + 1);	// unlock.
			} else {
				*tryagain = true;
				//printf("\tthread %d aborts, did not get lock on src %d.\n", id, srcboss);
			}
			atomicDec(&phores[srcboss], graph.nnodes + 1);	// unlock.
		}
	}
}

int main(int argc, char *argv[]) {
	float *mstwt, hmstwt = 0.0;
	int iteration = 0;
	Graph hgraph, graph;
	KernelConfig kconf;

	unsigned *partners, *phores;
	foru *eleminwts, *minwtcomponent;
	//bool *tryagain, htryagain;
	bool *processinnextiteration;
	unsigned *goaheadnodeofcomponent;

	double starttime, endtime;

	if (argc != 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}

	cudaGetLastError();

	hgraph.read(argv[1]);
	hgraph.cudaCopy(graph);
	//graph.print();

	kconf.setProblemSize(graph.nnodes);
	ComponentSpace cs(graph.nnodes);

	if (cudaMalloc((void **)&mstwt, sizeof(float)) != cudaSuccess) CudaTest("allocating mstwt failed");
	cudaMemcpy(mstwt, &hmstwt, sizeof(hmstwt), cudaMemcpyHostToDevice);	// mstwt = 0.
	//if (cudaMalloc((void **)&tryagain, sizeof(bool)) != cudaSuccess) CudaTest("allocating tryagain failed");

	if (cudaMalloc((void **)&eleminwts, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating eleminwts failed");
	if (cudaMalloc((void **)&minwtcomponent, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating minwtcomponent failed");
	if (cudaMalloc((void **)&partners, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating partners failed");
	if (cudaMalloc((void **)&phores, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating phores failed");
	if (cudaMalloc((void **)&processinnextiteration, graph.nnodes * sizeof(bool)) != cudaSuccess) CudaTest("allocating processinnextiteration failed");
	if (cudaMalloc((void **)&goaheadnodeofcomponent, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating goaheadnodeofcomponent failed");

	kconf.setMaxThreadsPerBlock();

	unsigned prevncomponents/*, innerprevncomponents*/, currncomponents = graph.nnodes;
	unsigned awhilebackncomps = currncomponents;

	printf("finding mst.\n");
	starttime = rtclock();

	do {
		++iteration;
		prevncomponents = currncomponents;
		dinit 		<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
		CudaTest("dinit failed");
		dfindelemin 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
		CudaTest("dfindelemin failed");
		dfindcompmin 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
		CudaTest("dfindcompmin failed");
		dfindcompmintwo <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
		CudaTest("dfindcompmintwo failed");

		/*do {
			htryagain = false;
			//innerprevncomponents = currncomponents;
			cudaMemcpy(tryagain, &htryagain, sizeof(htryagain), cudaMemcpyHostToDevice);
			dfindcompmin <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, tryagain);
			CudaTest("dfindcompmin failed");
			cudaMemcpy(&htryagain, tryagain, sizeof(htryagain), cudaMemcpyDeviceToHost);
			//currncomponents = cs.numberOfComponentsHost();
		} while (htryagain);// && currncomponents != innerprevncomponents);*/
		/*if (htryagain) {
			for (unsigned ii = 0; ii < graph.nnodes; ++ii) {
				dfindcompmin <<<1,1>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, tryagain, ii);
				CudaTest("dfindcompmin1x1 failed");
			}
		}*/

		currncomponents = cs.numberOfComponentsHost();
		cudaMemcpy(&hmstwt, mstwt, sizeof(hmstwt), cudaMemcpyDeviceToHost);
		//printf("\titeration %d, number of components = %d, mstwt = %.f.\n", iteration, currncomponents, hmstwt);
		//printf("\n");
		//cs.print();

		#define THRESHOLDCHANGE	10
		#define THRESHOLDNCOMPS	100
		if (iteration % THRESHOLDCHANGE == 0) {
			if (awhilebackncomps - currncomponents < 2*THRESHOLDCHANGE && currncomponents > THRESHOLDNCOMPS) {
				printf("shifting to sequential processing.\n");
				//getchar();
				do {
					++iteration;
					prevncomponents = currncomponents;
					for (unsigned ii = 0; ii < graph.nnodes; ++ii) {
						dinit          <<<1, 1>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, ii); CudaTest("dinit failed");
						dfindelemin    <<<1, 1>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, ii); CudaTest("dfindelemin failed");
						dfindcompmin   <<<1, 1>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, ii); CudaTest("dfindcompmin failed");
						dfindcompmintwo<<<1, 1>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, ii); CudaTest("dfindcompmintwo failed");
					}
					currncomponents = cs.numberOfComponentsHost();
					cudaMemcpy(&hmstwt, mstwt, sizeof(hmstwt), cudaMemcpyDeviceToHost);
					//printf("\titeration %d, number of components = %d, mstwt = %.f.\n", iteration, currncomponents, hmstwt);
					//printf("\n");
				} while (currncomponents != prevncomponents);
				break;
			}
			awhilebackncomps = currncomponents;
		}

	} while (currncomponents != prevncomponents);
	endtime = rtclock();
	
	printf("\tmstwt = %.f, iterations = %d.\n", hmstwt, iteration);
	printf("\truntime = %.3lf ms.\n", 1000 * (endtime - starttime));

	// cleanup left to the OS.

	return 0;
}
