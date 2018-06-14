#ifndef LSG_METISNODE
#define LSG_METISNODE

typedef struct MetisNode{

	union{
		int matched;		// only used in coarsening
		unsigned partition;	// used in partition and refinement
	} compress;

	int parent;				// used in coarsening
	int children[2];		// used in coarsening and refinement
	int lock;				// used in coarsening

	unsigned _weight;		// used everywhere

/*
	union{
		bool maybeBoundary;	// only using in refinement
		bool failedMatch;	// only used in coarsening
	} compress2;
*/

	bool maybeBoundary;		// only used in refinement
	bool failedMatch;		// only used in coarsening

	__device__ bool isMatched();

	__device__ __host__ void setMatched(int v);
	__device__ int getMatched();

	__device__ __host__ unsigned getPart();
	__device__ __host__ void setPart(unsigned p);
/*
	__device__ void setMatched(int v);
	__device__ int getMatched();

	__device__ unsigned getPart();
	__device__ void setPart(unsigned p);
*/

	__device__ void setParent(int p);
	__device__ int getParent();

	__device__ unsigned numChildren();

	__device__ void unlock();
	__device__ bool isLocked();
	//__device__ bool isLockedByMe(int tid);
	__device__ bool isLockedORMatchedANDLock(int tid);

	__device__ __host__ unsigned getWeight();
//	__device__ unsigned getWeight();

	__device__ __host__ void setWeight(unsigned weight);

	__device__ bool getmaybeBoundary();
	__device__ void setmaybeBoundary(bool val);

	__device__ void setFailedMatch();
	__device__ bool isFailedMatch();

	__device__ void initRefine(unsigned part, bool bound);

} MetisNode;

__device__ unsigned MetisNode::getWeight(){ return _weight; }

__device__ __host__ void MetisNode::setWeight(unsigned weight){ _weight = weight; }
//__device__ void MetisNode::setWeight(unsigned weight){ _weight = weight; }



__device__ void MetisNode::initRefine(unsigned part, bool bound) {
	compress.partition = part;
	maybeBoundary = bound;
	//compress2.maybeBoundary = bound;
}


__device__ bool MetisNode::getmaybeBoundary() { return maybeBoundary; }
__device__ void MetisNode::setmaybeBoundary(bool val){ maybeBoundary = val; }

//__device__ bool MetisNode::getmaybeBoundary() { return compress2.maybeBoundary; }
//__device__ void MetisNode::setmaybeBoundary(bool val){ compress2.maybeBoundary = val; }

__device__ unsigned MetisNode::getPart(){ return compress.partition; }

__device__ __host__ void MetisNode::setPart(unsigned p){ compress.partition = p; }
//__device__ void MetisNode::setPart(unsigned p){ compress.partition = p; }


__device__ bool MetisNode::isMatched(){ return (compress.matched >= 0); }
__device__ void MetisNode::setMatched(int v){ compress.matched = v; }
__device__ int MetisNode::getMatched(){ return compress.matched; }


__device__ void MetisNode::unlock(){ atomicExch(&lock, -1); }

__device__ bool isLockedByMe(MetisNode &node, int tid){ /*__threadfence();*/ return (node.lock == tid); }

__device__ bool MetisNode::isLocked(){/* __threadfence();*/ return (lock >= 0); }

/*
 * Checks wether a node is matched or locked by another thread
 * @arg tid - id of the thread that is trying to lock
 * @returns true if the node is matched or locked by another thread
 * @returns false if the thread is not matched and was/is locked by the current thread
 */
__device__ bool MetisNode::isLockedORMatchedANDLock(int tid){
	//__threadfence();
	//if(isMatched()) return true;

	int ret = atomicCAS(&lock, -1, tid);
	
	if(ret == -1 || ret == tid){
		//return false;
		return isMatched();
	}

	return true;
}

__device__ void MetisNode::setParent(int p){ parent = p; }
__device__ int MetisNode::getParent(){ return parent;}

__device__ unsigned MetisNode::numChildren(){ return children[1] >= 0 ? 2 : 1; }

__device__ void MetisNode::setFailedMatch(){ failedMatch = true; }
__device__ bool MetisNode::isFailedMatch(){ return failedMatch; }


//__device__ void MetisNode::setFailedMatch(){ compress2.failedMatch = true; }
//__device__ bool MetisNode::isFailedMatch(){ return compress2.failedMatch; }



#endif

