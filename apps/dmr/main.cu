/** Delaunay refinement -*- CUDA -*-
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
 * Refinement of an initial, unrefined Delaunay mesh to eliminate triangles
 * with angles < 30 degrees, using a variation of Chew's algorithm.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 */

#include "lonestargpu.h"

#define MINANGLE	30
#define PI		3.14159265358979323846	// from C99 standard.
#define FORD		float
#define DIMSTYPE	unsigned

#define INVALIDID	1234567890
#define MAXID		INVALIDID
#define TESTNBLOCKSFACTOR	4

#define ALLOCMULTIPLE	2	// alloc in multiples of this.
unsigned ALLOCFACTOR = 6;	// initial alloc factor.

  void next_line(std::ifstream& scanner) { 
    scanner.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
  } 

  void readNodes(std::string filename, FORD * &nodex, FORD * &nodey, unsigned &nnodes) {
    std::ifstream scanner(filename.append(".node").c_str());
    scanner >> nnodes;
    //next_line(scanner);
  
	nodex = (FORD *)malloc(nnodes * sizeof(FORD));
	nodey = (FORD *)malloc(nnodes * sizeof(FORD));

    for (size_t i = 0; i < nnodes; i++) {
      size_t index;
      FORD x; 
      FORD y;
      next_line(scanner);
      scanner >> index >> x >> y;
      nodex[index] = x;
      nodey[index] = y;
    } 
  }   
  
   void readTriangles(std::string basename, unsigned * &tnodes, unsigned &ntriangles, unsigned nnodes) {
	// bug on the placement of next_line identified by Molly O'Neil: fixed.
	unsigned ntrianglesone, ntrianglestwo;
	unsigned i, index, n1, n2, n3, row;
	std::string filename;
      
	filename = basename;
    std::ifstream scanner(filename.append(".ele").c_str());
	scanner >> ntrianglesone;

	filename = basename;
    std::ifstream scannerperimeter(filename.append(".poly").c_str());
	//next_line(scannerperimeter);
	scannerperimeter >> ntrianglestwo;

	ntriangles = ntrianglesone + ntrianglestwo;
	tnodes = (unsigned *)malloc(3 * ntriangles * sizeof(unsigned));

    for (i = 0; i < ntrianglesone; i++) {
	next_line(scanner);
      scanner >> index >> n1 >> n2 >> n3;
	row = 3 * index;
	tnodes[row + 0] = n1;
	tnodes[row + 1] = n2;
	tnodes[row + 2] = n3;
    }

    for (i = 0; i < ntrianglestwo; i++) {
	next_line(scannerperimeter);
      scannerperimeter >> index >> n1 >> n2;
	row = 3 * (ntrianglesone + index);
	tnodes[row + 0] = n1;
	tnodes[row + 1] = n2;
	tnodes[row + 2] = INVALIDID;
    }

  }

void optimizeone(unsigned ntriangles) {

}

__device__
FORD distanceSquare(FORD onex, FORD oney, FORD twox, FORD twoy) {
	FORD dx = onex - twox;
	FORD dy = oney - twoy;
	FORD dsq = dx * dx + dy * dy;
	return dsq;
}
__device__
FORD distanceSquare(unsigned one, unsigned two, FORD *nodex, FORD *nodey) {
	return distanceSquare(nodex[one], nodey[one], nodex[two], nodey[two]);
}
__device__
FORD distance(unsigned one, unsigned two, FORD *nodex, FORD *nodey) {
	return sqrtf(distanceSquare(one, two, nodex, nodey));
}
__device__
FORD radiusSquare(FORD centerx, FORD centery, unsigned tri, FORD *nodex, FORD *nodey, unsigned *tnodes) {
	unsigned row = 3 * tri;
	unsigned first = tnodes[row + 0];
	return distanceSquare(centerx, centery, nodex[first], nodey[first]);
}
__device__
bool checkbad(unsigned id, FORD *nodex, FORD *nodey, unsigned *tnodes, DIMSTYPE *obtuse, unsigned ntriangles) {
	//if (id < ntriangles) {
		unsigned row = 3 * id;
		DIMSTYPE dims = (tnodes[row + 2] == INVALIDID ? 2 : 3);

		for (unsigned ii = 0; ii < dims; ++ii) {
			unsigned curr = tnodes[row + ii];
			unsigned aa = tnodes[row + (ii + 1) % dims];
			unsigned bb = tnodes[row + (ii + 2) % dims];
			if (curr < ntriangles && aa < ntriangles && bb < ntriangles) {
				FORD vax = nodex[aa] - nodex[curr];
				FORD vay = nodey[aa] - nodey[curr];
				FORD vbx = nodex[bb] - nodex[curr];
				FORD vby = nodey[bb] - nodey[curr];
				FORD dp = vax * vbx + vay * vby;

				if (dp < 0) {
					// id is obtuse at point ii.
					obtuse[id] = ii;
				} else {
					FORD dsqaacurr = distanceSquare(aa, curr, nodex, nodey);
					FORD dsqbbcurr = distanceSquare(bb, curr, nodex, nodey);
					FORD c = dp * rsqrtf(dsqaacurr * dsqbbcurr);
					if (c > cos(MINANGLE * (PI / 180))) {
						return true;
					}
				}
			}
		}
	//}
	return false;
}
__global__
void dinit(FORD *nodex, FORD *nodey, unsigned *tnodes, bool *isbad, DIMSTYPE *obtuse, bool *isdel, unsigned nnodes, unsigned ntriangles) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < ntriangles) {
		obtuse[id] = 3;
		isbad[id] = checkbad(id, nodex, nodey, tnodes, obtuse, ntriangles);
		isdel[id] = false;
	}
}
__global__
void dverify(FORD *nodex, FORD *nodey, unsigned *tnodes, bool *isbad, bool *isdel, unsigned nnodes, unsigned ntriangles, bool *changed, unsigned *nchanged) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < ntriangles && !isdel[id] && isbad[id] ) {
		*changed = true;
		++*nchanged;
	}
}
__device__
unsigned adjacent(unsigned trione, unsigned tritwo, unsigned *tnodes, unsigned nnodes, unsigned ntriangles) {
	unsigned rowone = 3 * trione;
	unsigned rowtwo = 3 * tritwo;
	unsigned dimsone = (tnodes[rowone + 2] == INVALIDID ? 2 : 3);
	unsigned dimstwo = (tnodes[rowtwo + 2] == INVALIDID ? 2 : 3);
	unsigned ncommon = 0;
	unsigned firstmatch = 3;	// not adjacent.

	for (unsigned ii = 0; ii < dimsone; ++ii) {
		for (unsigned jj = 0; jj < dimstwo; ++jj) {
			if (tnodes[rowone + ii] == tnodes[rowtwo + jj]) {
				if (++ncommon == 2) {
					return firstmatch;
				} else {
					firstmatch = ii;
				}
			}
		}
	}
	return 3;	// not adjacent.
}
__global__
void dfindneighbors(FORD *nodex, FORD *nodey, unsigned *tnodes, unsigned *neighbors, DIMSTYPE *neighboredges, unsigned nnodes, unsigned ntriangles, unsigned nblocks, unsigned starttri, unsigned endtri) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned wpt = (ntriangles + nblocks * blockDim.x - 1) / nblocks / blockDim.x;
	unsigned wpt = (endtri - starttri + nblocks * blockDim.x - 1) / (nblocks * blockDim.x);	//1;
	unsigned start = starttri + id * wpt;
	unsigned end = start + wpt;

	for (unsigned tt = start; tt < end && tt < ntriangles; ++tt) {
		unsigned row = 3 * tt;
		unsigned iirow = 0;
		//for (unsigned ii = 0; ii < ntriangles; ++ii) {
		for (unsigned ii = starttri; ii < endtri; ++ii) {
			if (ii != tt) {
				unsigned commonedgestart = adjacent(tt, ii, tnodes, nnodes, ntriangles);
				if (commonedgestart < 3 && iirow < 3) {	// common edge, adjacent.
					neighbors[row + iirow] = ii;
					neighboredges[row + iirow] = commonedgestart;	// store the common edge for the first triangle, another thread will store it for the second triangle.
					++iirow;
				}
			}
		}
		// fill the remaining entries by invalid data.
		for (; iirow < 3; ++iirow) {
			neighbors[row + iirow] = INVALIDID;
			neighboredges[row + iirow] = 3;
		}
	}
}
__device__
unsigned getOpposite(unsigned centerelement, unsigned obtuse, unsigned *neighbors, DIMSTYPE *neighboredges, unsigned *tnodes, unsigned nnodes, unsigned ntriangles) {
	unsigned row = 3 * centerelement;
	DIMSTYPE dims = (tnodes[row + 2] == INVALIDID ? 2 : 3);
	DIMSTYPE commonedgepoint1 = (obtuse + 1) % dims;
	//unsigned commonedgepoint2 = (obtuse + 2) % dims;

	for (unsigned ii = 0; ii < 3; ++ii) {	// iterate over neighbors.
		DIMSTYPE nnedgestart = neighboredges[row + ii];
		if (nnedgestart == commonedgepoint1) {
			return neighbors[row + ii];
		}
	}
	return INVALIDID;
}

__device__
void getCenter(unsigned centerelement, FORD &centerx, FORD &centery, FORD *nodex, FORD *nodey, unsigned *tnodes, unsigned nnodes, unsigned ntriangles) {
	unsigned row = 3 * centerelement;
	DIMSTYPE dims = (tnodes[row + 2] == INVALIDID ? 2 : 3);

	unsigned aa = tnodes[row + 0];
	unsigned bb = tnodes[row + 1];
	unsigned cc = tnodes[row + 2];

	if (!(aa < ntriangles && bb < ntriangles && cc < ntriangles)) {
		centerx = centery = 0.0;
		return;
	}
	if (dims == 2) {
		centerx = (nodex[aa] + nodex[bb]) * 0.5;
		centery = (nodey[aa] + nodey[bb]) * 0.5;
		return;
	}
	FORD xxx = nodex[bb] - nodex[aa];
	FORD xxy = nodey[bb] - nodey[aa];
	FORD yyx = nodex[cc] - nodex[aa];
	FORD yyy = nodey[cc] - nodey[aa];
	
	FORD xxlen = distance(aa, bb, nodex, nodey);
	FORD yylen = distance(aa, cc, nodex, nodey);
	FORD cosine = (xxx * yyx + xxy * yyy) / (xxlen * yylen);
	FORD sinesq = 1.0 - cosine * cosine;
	FORD plen = yylen / xxlen;
	FORD ss = plen * cosine;
	FORD tt = plen * sinesq;
	FORD wp = (plen - cosine) / (2 * tt);
	FORD wb = 0.5 - (wp * ss);
	
	centerx = nodex[aa] * (1 - wb - wp) + nodex[bb] * wb + nodex[cc] * wp;
	centery = nodey[aa] * (1 - wb - wp) + nodey[bb] * wb + nodey[cc] * wp;
}
__device__
bool inCircumcircle(FORD xx, FORD yy, unsigned tri, FORD *nodex, FORD *nodey, unsigned *tnodes, unsigned nnodes, unsigned ntriangles) {
	// check if point (xx, yy) is in the circumcircle of tri.
	FORD centerx, centery;
	getCenter(tri, centerx, centery, nodex, nodey, tnodes, nnodes, ntriangles);
	FORD dd = distanceSquare(centerx, centery, xx, yy);
	return dd <= radiusSquare(centerx, centery, tri, nodex, nodey, tnodes);
}
__device__
unsigned addPoint(FORD xx, FORD yy, FORD *nodex, FORD *nodey, unsigned *pnnodes, unsigned &nnodes) {
	unsigned newpoint = *pnnodes; ++*pnnodes;	//atomicInc(pnnodes, MAXID);
	nodex[newpoint] = xx;
	nodey[newpoint] = yy;
	nnodes = newpoint;	// update.
	return newpoint;
}
__device__
void addPoint(FORD xx, FORD yy, FORD *nodex, FORD *nodey, unsigned newpoint) {
	nodex[newpoint] = xx;
	nodey[newpoint] = yy;
}
__device__
void initNeighbors(unsigned tri, unsigned *neighbors, DIMSTYPE *neighboredges) {
	unsigned row = 3 * tri;
	for (unsigned ii = 0; ii < 3; ++ii) {
		neighbors[row + ii] = INVALIDID;
		neighboredges[row + ii] = 3;
	}
}
__device__
unsigned addTriangle(unsigned point0, unsigned point1, unsigned point2, unsigned *tnodes, unsigned *pntriangles, unsigned &ntriangles, bool *isdel, DIMSTYPE *obtuse, unsigned *neighbors, DIMSTYPE *neighboredges) {
	unsigned newtriid = atomicInc(pntriangles, MAXID);
	unsigned newrow = 3 * newtriid;
	tnodes[newrow + 0] = point0;
	tnodes[newrow + 1] = point1;
	tnodes[newrow + 2] = point2;
	initNeighbors(newtriid, neighbors, neighboredges);
	isdel[newtriid] = false;
	obtuse[newtriid] = 3;
	ntriangles = newtriid;	// update.
	return newtriid;
}
__device__
void copyNeighbors(unsigned to, unsigned from, unsigned *neighbors, DIMSTYPE *neighboredges) {
	unsigned torow = 3 * to;
	unsigned fromrow = 3 * from;
	for (unsigned ii = 0; ii < 3; ++ii) {
		neighbors[torow + ii] = neighbors[fromrow + ii];
		neighboredges[torow + ii] = neighboredges[fromrow + ii];	// ???
	}
}
__device__
bool updateNeighbor(unsigned of, unsigned oldn, unsigned newn, unsigned *neighbors, unsigned *tnodes) {
	unsigned row = 3 * of;
	DIMSTYPE dims = (tnodes[row + 2] == INVALIDID ? 2 : 3);
	for (unsigned ii = 0; ii < dims; ++ii) {
		if (neighbors[row + ii] == oldn) {
			neighbors[row + ii] = newn;
			return true;
		}
	}
	// no need to update neighboredges, as the index won't change.
	return false;
}
void *mycudarealloc(void *oldptr, unsigned oldsize, unsigned newsize) {
	void *newptr;
	if (cudaMalloc((void **)&newptr, newsize) != cudaSuccess) CudaTest("allocating newptr failed");
	cudaMemcpy(newptr, oldptr, oldsize, cudaMemcpyDeviceToDevice);
	cudaFree(oldptr);
	return newptr;
}
//GPU lock-free synchronization function
__device__ 
void __gpu_sync(unsigned goalVal, volatile unsigned *Arrayin, volatile unsigned *Arrayout) {
	// thread ID in a block
	unsigned tid_in_blk = threadIdx.x * blockDim.y + threadIdx.y;
	unsigned nBlockNum = gridDim.x * gridDim.y;
	unsigned bid = blockIdx.x * gridDim.y + blockIdx.y;
	// only thread 0 is used for synchronization
	if (tid_in_blk == 0) {
		Arrayin[bid] = goalVal;
		__threadfence();
	}
	if (bid == 0) {
		if (tid_in_blk < nBlockNum) {
			while (Arrayin[tid_in_blk] != goalVal){
				//Do nothing here
			}
		}
		__syncthreads();
		if (tid_in_blk < nBlockNum) {
			Arrayout[tid_in_blk] = goalVal;
			__threadfence();
		}
	}
	if (tid_in_blk == 0) {
		while (Arrayout[bid] != goalVal) {
			//Do nothing here
		}
	}
	__syncthreads();
}


__device__
void globalsyncthreads(unsigned &blockcount, volatile unsigned *go) {
	unsigned tt;
	if (threadIdx.x == 0) {
		tt = gridDim.x - 1;
		if (tt == atomicInc(&blockcount, tt)) {
			*go = 1;
			__threadfence_block();
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		while (*go != 1) {
			;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		tt = gridDim.x - 1;
		if (tt == atomicInc(&blockcount, tt)) {
			*go = 0;
			__threadfence_block();
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		while (*go != 0) {
			;
		}
	}
	__syncthreads();
}
__global__
void countbad(bool *isbad, unsigned ntriangles, unsigned *nbad, unsigned goal, volatile unsigned *arrayin, volatile unsigned *arrayout, unsigned *blockcount) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nthreads = blockDim.x * gridDim.x;
	unsigned wpt = (ntriangles + nthreads - 1) / nthreads;
	unsigned start = id*wpt;
	unsigned end = start + wpt;
	__shared__ unsigned tcount[BLOCKSIZE];

	unsigned imemyself = threadIdx.x;
	tcount[imemyself] = 0;
	for (unsigned ii = start; ii < end; ++ii) {
		if (ii < ntriangles && isbad[ii]) {
			++tcount[imemyself];
		}
	}
	__syncthreads();

       for (unsigned s = blockDim.x / 2; s; s >>= 1) {
                if (imemyself < s) {
                        tcount[imemyself] += tcount[imemyself + s];
                }
                __syncthreads();
        }
        __syncthreads();

	if (imemyself == 0) {
		blockcount[blockIdx.x] = tcount[0];
		__threadfence();
	}
	__gpu_sync(++goal, arrayin, arrayout);
	if (id == 0) {
		unsigned lcount = 0;
		for (unsigned ii = 0; ii < gridDim.x; ++ii) {
			lcount += blockcount[ii];
		}
		*nbad = lcount;
	}
}
#define DEBUGCHECK(ii)	if (ii >= SMALLSIZE) { printf("ERROR %s: %d.\n", #ii, ii);}
#define DEBUGCHECK4(ii)	if (ii >= 4*SMALLSIZE) { printf("ERROR %s: %d.\n", #ii, ii);}
#define DEBUGCHECKN(ii, N)	if (ii >= N) { printf("ERROR %s: %d.\n", #ii, ii);}
#define MAXITR	10
__global__
__launch_bounds__(BLOCKSIZE, TESTNBLOCKSFACTOR)
void drefine(FORD *nodex, FORD *nodey, unsigned *tnodes, unsigned *neighbors, DIMSTYPE *neighboredges, bool *isbad, DIMSTYPE *obtuse, bool *isdel, unsigned *pnnodes, unsigned *pntriangles, bool *changed, unsigned starttri, unsigned endtri, unsigned nblocks, unsigned *owner, unsigned *successful, unsigned *aborted, unsigned *blockcount, volatile unsigned *go, unsigned goal, volatile unsigned *arrayin, volatile unsigned *arrayout) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nthreads = blockDim.x * nblocks;
	unsigned wpt = (endtri - starttri + nthreads - 1) / nthreads;	//1;
	unsigned start = starttri + id * wpt;
	unsigned end = start + wpt;
	unsigned nnodes = *pnnodes;
	unsigned ntriangles = *pntriangles;

	unsigned centerelement = 0, row = 0;
	DIMSTYPE ceobtuse = 3, dims = 3;
	FORD centerx = 0.0, centery = 0.0;

	bool lchanged = false;
		#define SMALLSIZE	64
		unsigned frontier[SMALLSIZE], iifrontier = 0;
		unsigned pre[SMALLSIZE], iipre = 0;
		unsigned post[SMALLSIZE], iipost = 0;
		unsigned connections[4 * SMALLSIZE], iiconnections = 0;	// edgesrc, edgedst, triangleone, triangletwo.
	
	for (unsigned tt = start; tt < end; ++tt) {
	    if (tt < ntriangles && !isdel[tt] && isbad[tt]) {
		
		iifrontier = iipre = iipost = iiconnections = 0;
	
		// cavity.initialize(tt);
		centerelement = tt;
		ceobtuse = obtuse[centerelement];
		unsigned itr = 0;
		while (ceobtuse < 3 && centerelement < ntriangles && ++itr < MAXITR) {	// while it is obtuse.
			centerelement = getOpposite(centerelement, ceobtuse, neighbors, neighboredges, tnodes, nnodes, ntriangles);
			if (centerelement < ntriangles) {
				ceobtuse = obtuse[centerelement];
			}
		}
		if (centerelement >= ntriangles || isdel[centerelement]) {
			centerelement = tt;
			ceobtuse = obtuse[centerelement];
		}
		getCenter(centerelement, centerx, centery, nodex, nodey, tnodes, nnodes, ntriangles);
		
		pre[iipre++] = centerelement;
		frontier[iifrontier++] = centerelement;
		//DEBUGCHECK(iipre);
		
		// cavity.build();
		while (iifrontier > 0) {
			unsigned curr = frontier[--iifrontier];
			unsigned row = 3 * curr;
			DIMSTYPE dims = (tnodes[row + 2] == INVALIDID ? 2 : 3);

			for (unsigned ii = 0; ii < dims; ++ii) {
				//expand(curr, neighbors[row + ii]);
				unsigned next = neighbors[row + ii];
				if (next >= ntriangles) {
					break;
				}
				if (isdel[next]) {
					continue;
				}
				unsigned nextrow = 3 * next;
				unsigned nextdims = (tnodes[nextrow + 2] == INVALIDID ? 2 : 3);
				if (!(dims == 2 && nextdims == 2 && next != centerelement) && inCircumcircle(centerx, centery, next, nodex, nodey, tnodes, nnodes, ntriangles)) {
					// isMember says next is part of the cavity, and we're not the second
					// segment encroaching on this cavity
					if (nextdims == 2 && dims != 2) {
						// is segment, and we are encroaching.
						iifrontier = iipre = iipost = iiconnections = 0;
						centerelement = next;
						ceobtuse = obtuse[centerelement];
						itr = 0;
						while (ceobtuse < 3 && centerelement < ntriangles && ++itr < MAXITR) {
							centerelement = getOpposite(centerelement, ceobtuse, neighbors, neighboredges, tnodes, nnodes, ntriangles);
							if (centerelement < ntriangles) {
								ceobtuse = obtuse[centerelement];
							}
						}
						if (centerelement >= ntriangles || isdel[centerelement]) {
							centerelement = next;
							ceobtuse = obtuse[centerelement];
						}
						getCenter(centerelement, centerx, centery, nodex, nodey, tnodes, nnodes, ntriangles);
						pre[iipre++] = centerelement;
						frontier[iifrontier++] = centerelement;
						//DEBUGCHECK(iipre);
					} else {
						unsigned jj;
						for (jj = 0; jj < iipre; ++jj) {
							if (pre[jj] == next) {
								break;
							}
						}
						if (jj == iipre) {
							pre[iipre++] = next;
							frontier[iifrontier++] = next;
						}
						//DEBUGCHECK(iipre);
					}
				} else {
					// not a member
					// add the common edge between curr and next to connections if doesn't already exist.
					DIMSTYPE cestart = neighboredges[row + ii];	// see definition of next above.
					if (cestart >= 3) {
						continue;
					}
					unsigned connpt1 = tnodes[row + cestart];
					unsigned connpt2 = tnodes[row + (cestart + 1) % dims];
					
					unsigned jj;
					for (jj = 0; jj < iiconnections; jj += 4) {
						if (connections[jj] == connpt1 && connections[jj + 1] == connpt2) {
							break;
						}
					}
					if (jj == iiconnections) {
						connections[iiconnections++] = connpt1;
						connections[iiconnections++] = connpt2;
						connections[iiconnections++] = curr;
						connections[iiconnections++] = next;
						//DEBUGCHECK4(iiconnections);
					}
				}
			}
		}
		// mark the triangles in the cavity.
		for (unsigned ii = 0; ii < iipre; ++ii) {
			unsigned cavtri = pre[ii];
			if (cavtri < endtri && cavtri >= starttri) {
				owner[cavtri] = id;
			}
		}
	    }
		//__syncthreads();
		//__threadfence();
		//globalsyncthreads(*blockcount, go);
		__gpu_sync(++goal, arrayin, arrayout);
		
		bool backoff = false;
	    if (tt < ntriangles && !isdel[tt] && isbad[tt]) {
		// go over your triangles and see if they contain your id.
		if (!backoff) {
			for (unsigned ii = 0; ii < iipre; ++ii) {
				unsigned cavtri = pre[ii];
				if (owner[cavtri] < id) {	// cavity overlap and the other thread has priority!
					backoff = true;
					break;
				} else if (owner[cavtri] > id) {	// cavity overlap but you have the priority.
					owner[cavtri] = id;	// mark it yours: due to this write, we require another checking phase.
				}
			}
		}
		//__syncthreads();
		//__threadfence();
	    }
		//globalsyncthreads(*blockcount, go);
		__gpu_sync(++goal, arrayin, arrayout);

	    if (tt < ntriangles && !isdel[tt] && isbad[tt]) {
		// once again go over your triangles and see if they contain your id.
		if (!backoff) {
			for (unsigned ii = 0; ii < iipre; ++ii) {
				unsigned cavtri = pre[ii];
				if (owner[cavtri] != id) {	// cavity overlap.
					backoff = true;
					break;
				}
			}
		}
		//__syncthreads();

		if (backoff) {
			lchanged = true;
			++*aborted;
			continue;
		}
		++*successful;
		// cavity.update(): create the new cavity based on the data of the old cavity.
		row = 3 * centerelement;
		dims = (tnodes[row + 2] == INVALIDID ? 2 : 3);
		unsigned newpoint = addPoint(centerx, centery, nodex, nodey, pnnodes, nnodes);

		if (dims == 2) {	// we built around a segment.
			// create two segments (as triangles).
			unsigned newtriid1 = addTriangle(newpoint, tnodes[row + 0], INVALIDID, tnodes, pntriangles, ntriangles, isdel, obtuse, neighbors, neighboredges);
			unsigned newtriid2 = addTriangle(newpoint, tnodes[row + 1], INVALIDID, tnodes, pntriangles, ntriangles, isdel, obtuse, neighbors, neighboredges);
			// update triangles' neighbors: neighbors of the new triangles (segments) are the same as those of the previous segment?
			copyNeighbors(newtriid1, centerelement, neighbors, neighboredges);
			copyNeighbors(newtriid2, centerelement, neighbors, neighboredges);

			post[iipost++] = newtriid1;
			post[iipost++] = newtriid2;
			//DEBUGCHECK(iipost);
		}
	
		for (unsigned ii = 0; ii < iiconnections; ii += 4) {
			unsigned connpt1 = connections[ii + 0];
			unsigned connpt2 = connections[ii + 1];
			unsigned connsrc = connections[ii + 2];
			unsigned conndst = connections[ii + 3];
			unsigned newtri = addTriangle(newpoint, connpt1, connpt2, tnodes, pntriangles, ntriangles, isdel, obtuse, neighbors, neighboredges);

			unsigned jj;
			for (jj = 0; jj < iipre; ++jj) {
				if (pre[jj] == conndst) {
					break;
				}
			}
			unsigned newconn = (jj == iipre ? conndst : connsrc);
			// newtri and newconn are triangles, and their common edge is (connpt1, connpt2).
			// thus they are adjacent; their neighbors need to be updated.
			unsigned newrow = 3 * newtri;
			unsigned iineighbor = 0;
			neighbors[newrow + iineighbor] = newconn;
			neighboredges[newrow + iineighbor] = 1;	// since connpt1 is point1 (newpoint is point0).
			++iineighbor;
			for (unsigned jj = 0; jj < iipost; ++jj) {
				DIMSTYPE commonedgestart = adjacent(post[jj], newtri, tnodes, nnodes, ntriangles);
				if (commonedgestart < 3) {
					if (iineighbor < 3) {
						//DEBUGCHECKN(iineighbor, 3);
						neighbors[newrow + iineighbor] = post[jj];
						neighboredges[newrow + iineighbor] = commonedgestart;
						++iineighbor;
					}
					updateNeighbor(post[jj], newconn, newtri, neighbors, tnodes);	// update neighbor of post[jj] from newconn to newtri, no need to change neighboredges.
				}
			}
			if (iipost < SMALLSIZE) {
				post[iipost++] = newtri;
				//DEBUGCHECK(iipost);
			}
		}
		// remove triangles from pre.
		for (unsigned ii = 0; ii < iipre; ++ii) {
			unsigned tri = pre[ii];
			//DEBUGCHECKN(tri, ntriangles);
			isdel[tri] = true;
		}
		
		// add triangles from post, mark the bad triangles.
		// triangles are already added using addTriangle(), simply mark the bad ones.
		for (unsigned ii = 0; ii < iipost; ++ii) {
			unsigned tri = post[ii];
			//DEBUGCHECKN(tri, 5000000);
			if (tri < ntriangles) {
				obtuse[tri] = 3;
				isbad[tri] = checkbad(tri, nodex, nodey, tnodes, obtuse, ntriangles);
				lchanged |= isbad[tri];
			}
		}
		// add neighborhood information for the new triangles: already added using updateNeighbor.
	    }
	}
	if (lchanged) {
		*changed = true;
	}
}
int main(int argc, char *argv[]) {
	unsigned int ntriangles, nnodes, *pnnodes, *pntriangles;
	bool *changed, hchanged;
	unsigned *nchanged, hnchanged, *owner;
	FORD *nodex, *nodey, *hnodex, *hnodey;
	unsigned *tnodes, *htnodes;
	unsigned *neighbors; DIMSTYPE *neighboredges;
	bool *isbad, *isdel;
	DIMSTYPE *obtuse;
	int iteration = 0;
	unsigned hsuccessful = 0, haborted = 0, *successful, *aborted;
	unsigned *blockcount, intzero = 0;
	volatile unsigned *go;
	volatile unsigned *arrayin, *arrayout;
	unsigned *bcount, *nbad, hnbad;
	KernelConfig kconf;

	clock_t starttime, endtime;
	int runtime;

	std::string str;

	//cudaDeviceProp deviceProp;
	//cudaGetDeviceProperties(&deviceProp, 0);
	//NBLOCKS = deviceProp.multiProcessorCount;

	cudaFuncSetCacheConfig(drefine, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(countbad, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(dfindneighbors, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(dverify, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(dinit, cudaFuncCachePreferL1);

	if (argc != 2) {
		printf("Usage: %s <basefilename>\n", argv[0]);
		exit(1);
	}

	cudaGetLastError();

	std::cout << "reading graphs...\n";
	readNodes(argv[1], hnodex, hnodey, nnodes);
	std::cout << "\t" << nnodes << " nodes\n";
	readTriangles(argv[1], htnodes, ntriangles, nnodes);
	std::cout << "\t" << ntriangles << " triangles.\n";
	
	kconf.setProblemSize(ntriangles);
	kconf.setNumberOfBlockThreads(256);
	//FACTOR = (ntriangles + BLOCKSIZE * NBLOCKS - 1) / (BLOCKSIZE * NBLOCKS);


	printf("optimizing.\n");
	optimizeone(ntriangles);

	unsigned curralloc = ALLOCFACTOR * ntriangles, currsizenodes = ALLOCFACTOR * nnodes;

	if (cudaMalloc((void **)&nodex, ALLOCFACTOR * nnodes * sizeof(FORD)) != cudaSuccess) CudaTest("allocating nodex failed");
	if (cudaMalloc((void **)&nodey, ALLOCFACTOR * nnodes * sizeof(FORD)) != cudaSuccess) CudaTest("allocating nodey failed");
	if (cudaMalloc((void **)&tnodes, ALLOCFACTOR * 3 * ntriangles * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating tnodes failed");

	cudaMemcpy(nodex, hnodex, nnodes * sizeof(FORD), cudaMemcpyHostToDevice);
	cudaMemcpy(nodey, hnodey, nnodes * sizeof(FORD), cudaMemcpyHostToDevice);
	cudaMemcpy(tnodes, htnodes, 3 * ntriangles * sizeof(unsigned), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&neighbors, ALLOCFACTOR * 3 * ntriangles * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating neighbors failed");
	if (cudaMalloc((void **)&neighboredges, ALLOCFACTOR * 3 * ntriangles * sizeof(DIMSTYPE)) != cudaSuccess) CudaTest("allocating neighboredges failed");
	//printf("finding neighboring triangles.\n");
	//unsigned nblocks = NBLOCKS * FACTOR;
	unsigned ntriperit = kconf.getNumberOfSMs() * kconf.getNumberOfBlockThreads();
	unsigned ntriit = kconf.getProblemSize() / ntriperit;
	//unsigned ntriit = FACTOR;
	//unsigned ntriperit = NBLOCKS * BLOCKSIZE;

	starttime = clock();
	for (unsigned ii = 0; ii < ntriit; ++ii) {
		printf("finding neighbors: %3d%% complete.\r", (int)(ii*ntriperit*100.0 / ntriangles));
		//printf("finding neighbors: iteration=%d, start=%d, end=%d.\n", ii, ii * ntriperit, (ii + 1) * ntriperit);
		//dfindneighbors<<<NBLOCKS, BLOCKSIZE>>> (nodex, nodey, tnodes, neighbors, neighboredges, nnodes, ntriangles, NBLOCKS, 0, ntriangles);
		dfindneighbors<<<kconf.getNumberOfSMs(), kconf.getNumberOfBlockThreads()>>> (nodex, nodey, tnodes, neighbors, neighboredges, nnodes, ntriangles, kconf.getNumberOfSMs(), ii * ntriperit, (ii + 1) * ntriperit);
		CudaTest("find neighbors failed");
	}
	endtime = clock();
	printf("\n");
	printf("findneighbors took %u ms.\n", (int)(1000.0f * (endtime - starttime) / CLOCKS_PER_SEC));

	if (cudaMalloc((void **)&isbad, ALLOCFACTOR * ntriangles * sizeof(bool)) != cudaSuccess) CudaTest("allocating isbad failed");
	if (cudaMalloc((void **)&obtuse, ALLOCFACTOR * ntriangles * sizeof(DIMSTYPE)) != cudaSuccess) CudaTest("allocating obtuse failed");
	if (cudaMalloc((void **)&isdel, ALLOCFACTOR * ntriangles * sizeof(bool)) != cudaSuccess) CudaTest("allocating isdel failed");
	if (cudaMalloc((void **)&owner, ALLOCFACTOR * ntriangles * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating owner failed");

	printf("init.\n");
	dinit <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (nodex, nodey, tnodes, isbad, obtuse, isdel, nnodes, ntriangles);
	CudaTest("initialization failed");
		/*bool *hisbad = (bool *)malloc(ntriangles * sizeof(bool));
		cudaMemcpy(hisbad, isbad, ntriangles * sizeof(bool), cudaMemcpyDeviceToHost);
		unsigned nbad = 0;
		for (unsigned ii = 0; ii < ntriangles; ++ii) {
			if (hisbad[ii]) nbad++;
		}
		std::cout << nbad << " bad triangles.\n";*/

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");
	if (cudaMalloc((void **)&nchanged, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nchanged failed");
	if (cudaMalloc((void **)&pnnodes, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating pnnodes failed");
	if (cudaMalloc((void **)&pntriangles, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating pntriangles failed");

	cudaMemcpy(pnnodes, &nnodes, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(pntriangles, &ntriangles, sizeof(unsigned), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&successful, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating successful failed");
	if (cudaMalloc((void **)&aborted, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating aborted failed");
	if (cudaMalloc((void **)&blockcount, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating blockcount failed");
	if (cudaMalloc((void **)&go, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating go failed");
	cudaMemcpy(blockcount, &intzero, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)go, &intzero, sizeof(unsigned), cudaMemcpyHostToDevice);


	unsigned nblockfactor = TESTNBLOCKSFACTOR;	//(ntriangles < 1000000 ? 7 : (ntriangles < 10000000 ? 31 : 61));	// for 250k.2, use 7, for r1M use 31, for r5M use 61.
	unsigned nblocks = kconf.getNumberOfSMs() * nblockfactor;
	unsigned blocksize = kconf.getNumberOfBlockThreads();
	bool hlchanged;

	if (cudaMalloc((void **)&arrayin, nblocks*sizeof(volatile unsigned)) != cudaSuccess) CudaTest("allocating arrayin failed");
	if (cudaMalloc((void **)&arrayout, nblocks*sizeof(volatile unsigned)) != cudaSuccess) CudaTest("allocating arrayout failed");
	if (cudaMalloc((void **)&bcount, nblocks*sizeof(unsigned)) != cudaSuccess) CudaTest("allocating blockcount failed");
	if (cudaMalloc((void **)&nbad, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nbad failed");
	curralloc = ALLOCFACTOR * ntriangles;
	
	printf("solving.\n");
	starttime = clock();

	do {
		++iteration;
		//printf("iteration %d: ntriangles=%d, nnodes=%d.\n", iteration, ntriangles, nnodes);
		unsigned orintriangles = ntriangles;
		hchanged = false;
		cudaMemcpy(changed, &hchanged, sizeof(bool), cudaMemcpyHostToDevice);
		hsuccessful = haborted = 0;
		cudaMemcpy(successful, &hsuccessful, sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemcpy(aborted, &haborted, sizeof(unsigned), cudaMemcpyHostToDevice);

		countbad <<<nblocks, blocksize>>> (isbad, ntriangles, nbad, 1000 + iteration, arrayin, arrayout, bcount);
		CudaTest("countbad failed");
		cudaMemcpy(&hnbad, nbad, sizeof(unsigned), cudaMemcpyDeviceToHost);
		//printf("iteration %d: nbad = %d.\n", iteration, hnbad);
		if (ntriangles + 2 * hnbad > curralloc) {	// here 2 is the no of new triangles added for each bad triangle.
			nodex = (FORD *)mycudarealloc(nodex, currsizenodes*sizeof(FORD), ALLOCMULTIPLE*currsizenodes*sizeof(FORD));
			nodey = (FORD *)mycudarealloc(nodey, currsizenodes*sizeof(FORD), ALLOCMULTIPLE*currsizenodes*sizeof(FORD));
			currsizenodes = ALLOCMULTIPLE*currsizenodes;

			tnodes = (unsigned *)mycudarealloc(tnodes, 3*curralloc*sizeof(unsigned), ALLOCMULTIPLE*3*curralloc*sizeof(unsigned));
			neighbors = (unsigned *)mycudarealloc(neighbors, 3*curralloc*sizeof(unsigned), ALLOCMULTIPLE*3*curralloc*sizeof(unsigned));
			neighboredges = (unsigned *)mycudarealloc(neighboredges, 3*curralloc*sizeof(DIMSTYPE), ALLOCMULTIPLE*3*curralloc*sizeof(DIMSTYPE));

			isbad = (bool *)mycudarealloc(isbad, curralloc*sizeof(bool), ALLOCMULTIPLE*curralloc*sizeof(bool));
			obtuse = (DIMSTYPE *)mycudarealloc(obtuse, curralloc*sizeof(DIMSTYPE), ALLOCMULTIPLE*curralloc*sizeof(DIMSTYPE));
			isdel = (bool *)mycudarealloc(isdel, curralloc*sizeof(bool), ALLOCMULTIPLE*curralloc*sizeof(bool));
			owner = (unsigned *)mycudarealloc(owner, curralloc*sizeof(unsigned), ALLOCMULTIPLE*curralloc*sizeof(unsigned));

			curralloc *= ALLOCMULTIPLE;
			printf("\t\tallocating memory to %d.\n", curralloc);
		}
		
		ntriperit = ntriangles;
		ntriit = (ntriangles + ntriperit - 1) / ntriperit;	//1;	//FACTOR;
		for (unsigned ii = 0; ii < ntriit; ++ii) {
			//printf("solving: inner iteration=%d, ntriangles=%d, nnodes=%d.\n", ii, ntriangles, nnodes);
			drefine <<<nblocks, blocksize>>> (nodex, nodey, tnodes, neighbors, neighboredges, isbad, obtuse, isdel, pnnodes, pntriangles, changed, ii * ntriperit, (ii + 1)*ntriperit, nblocks, owner, successful, aborted, blockcount, go, iteration, arrayin, arrayout);
			CudaTest("solving failed");
			cudaMemcpy(&nnodes, pnnodes, sizeof(unsigned), cudaMemcpyDeviceToHost);
			cudaMemcpy(&ntriangles, pntriangles, sizeof(unsigned), cudaMemcpyDeviceToHost);
			cudaMemcpy(&hsuccessful, successful, sizeof(unsigned), cudaMemcpyDeviceToHost);
			cudaMemcpy(&haborted, aborted, sizeof(unsigned), cudaMemcpyDeviceToHost);
			//printf("\tsuccessful=%d, aborted=%d.\n", hsuccessful, haborted);
			cudaMemcpy(&hlchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost);
			hchanged |= hlchanged;
		}	
		if (hchanged && orintriangles == ntriangles) {
			nblocks	= blocksize = 1;
		} else {
			nblocks = kconf.getNumberOfSMs() * nblockfactor;
			blocksize = kconf.getNumberOfBlockThreads();
		}
	} while (hchanged);
	endtime = clock();

	printf("verifying...\n");
	hnchanged = 0;
	cudaMemcpy(nchanged, &hnchanged, sizeof(unsigned), cudaMemcpyHostToDevice);
	hchanged = false;
	cudaMemcpy(changed, &hchanged, sizeof(bool), cudaMemcpyHostToDevice);
	dverify <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (nodex, nodey, tnodes, isbad, isdel, nnodes, ntriangles, changed, nchanged);
	CudaTest("verification failed");
	cudaMemcpy(&hchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hnchanged, nchanged, sizeof(unsigned), cudaMemcpyDeviceToHost);
	if (hchanged) {
		printf("verification failed: bad triangles exist: %d.\n", hnchanged);
	} else {
		printf("verification succeeded: 0 bad triangles exist.\n");
	}

	printf("iterations = %d.\n", iteration);
	runtime = (int) (1000.0f * (endtime - starttime) / CLOCKS_PER_SEC);
	printf("%d ms.\n", runtime);
	

	// cleanup left to the OS.

	return 0;
}
