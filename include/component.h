struct ComponentSpace {
	ComponentSpace(unsigned nelements);
	
	__device__ unsigned numberOfElements();
	__device__ unsigned numberOfComponents();
	__device__ bool isBoss(unsigned element);
	__device__ unsigned find(unsigned lelement, bool compresspath = true);
	__device__ bool unify(unsigned one, unsigned two);
	__device__ void print1x1();
	__host__   void print();

	void allocate();
	void init();
	unsigned numberOfComponentsHost();

	unsigned nelements;
	unsigned *ncomponents,			// number of components.
		 *complen, 			// lengths of components.
		 *ele2comp;			// components of elements.
};
ComponentSpace::ComponentSpace(unsigned nelements) {
	this->nelements = nelements;

	allocate();
	init();
}
__device__ void ComponentSpace::print1x1() {
	printf("\t\t-----------------\n");
	for (unsigned ii = 0; ii < nelements; ++ii) {
		printf("\t\t%d -> %d\n", ii, ele2comp[ii]);
	}	
	printf("\t\t-----------------\n");
}
__global__ void print1x1(ComponentSpace cs) {
	cs.print1x1();
}
__host__ void ComponentSpace::print() {
	::print1x1<<<1,1>>>(*this);
	CudaTest("cs.print1x1 failed");
}
__device__ unsigned ComponentSpace::numberOfElements() {
	return nelements;
}
__device__ unsigned ComponentSpace::numberOfComponents() {
	return *ncomponents;
}
unsigned ComponentSpace::numberOfComponentsHost() {
	unsigned hncomponents = 0;
	cudaMemcpy(&hncomponents, ncomponents, sizeof(unsigned), cudaMemcpyDeviceToHost);
	return hncomponents;
}
void ComponentSpace::allocate() {
	if (cudaMalloc((void **)&ncomponents, 1 * sizeof(unsigned)) != cudaSuccess) 
		CudaTest("allocating ncomponents failed");
	if (cudaMalloc((void **)&complen, nelements * sizeof(unsigned)) != cudaSuccess) 
		CudaTest("allocating complen failed");
	if (cudaMalloc((void **)&ele2comp, nelements * sizeof(unsigned)) != cudaSuccess) 
		CudaTest("allocating ele2comp failed");
}
__global__ void dinitcs(unsigned nelements, unsigned *complen, unsigned *ele2comp) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < nelements) {
		//elements[id] 	= id;
		complen[id]	= 1;
		ele2comp[id]	= id;
	}
}
void ComponentSpace::init() {
	// init the elements.
	unsigned blocksize = MAXBLOCKSIZE;	////
	unsigned nblocks = (nelements + blocksize - 1) / blocksize;
	dinitcs<<<nblocks, blocksize>>>(nelements, complen, ele2comp);
	CudaTest("dinitcs failed");

	// init number of components.
	cudaMemcpy(ncomponents, &nelements, sizeof(unsigned), cudaMemcpyHostToDevice);
}
__device__ bool ComponentSpace::isBoss(unsigned element) {
	return ele2comp[element] == element;
}
__device__ unsigned ComponentSpace::find(unsigned lelement, bool compresspath/*= true*/) {
	// do we need to worry about concurrency in this function?
	// for other finds, no synchronization necessary as the data-structure is a tree.
	// for other unifys, synchornization is not required considering that unify is going to affect only bosses, while find is going to affect only non-bosses.
	unsigned element = lelement;
	while (isBoss(element) == false) {
		element = ele2comp[element];
	}
	if (compresspath) ele2comp[lelement] = element;	// path compression.
	return element;
}
__device__ bool ComponentSpace::unify(unsigned one, unsigned two) {
	// if the client makes sure that one component is going to get unified as a source with another destination only once, then synchronization is unnecessary.
	// while this is true for MST, due to load-balancing in if-block below, a node may be source multiple times.
	// if a component is source in one thread and destination is another, then it is okay for MST.
    do {
	unsigned onecomp = find(one, false);
	unsigned twocomp = find(two, false);

	if (onecomp == twocomp) return false;

	unsigned boss = twocomp;
	unsigned subordinate = onecomp;
	//if (complen[onecomp] > complen[twocomp]) {	// one is larger, make it the representative: can create cycles.
	if (boss < subordinate) {			// break cycles by id.
		boss = onecomp;
		subordinate = twocomp;
	}
	// merge subordinate into the boss.
	//ele2comp[subordinate] = boss;
	unsigned oldboss = atomicExch(&ele2comp[subordinate], boss);
	if (oldboss != subordinate) {	// someone else updated the boss.
		// we need not restore the ele2comp[subordinate], as union-find ensures correctness and complen of subordinate doesn't matter.
		one = oldboss;
		two = boss;
	} else {
		dprintf("\t\tunifying %d -> %d (%d)\n", subordinate, boss);
		atomicAdd(&complen[boss], complen[subordinate]);
		//complen[boss] += complen[subordinate];
		// complen[subordinate] doesn't matter now, since find() will find its boss.
	
		// a component has reduced.
		unsigned ncomp = atomicSub(ncomponents, 1);
		//atomicDec(ncomponents, nelements);
		dprintf("\t%d: ncomponents = %d\n", threadIdx.x, ncomp);
		return true;
	}
    } while (true);
}
