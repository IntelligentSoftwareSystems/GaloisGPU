INCLUDES=-I../../include
CUDA_ARCH=sm_20

la: la.cu ../../include/lockarray.h
	nvcc -O3 -arch $(CUDA_ARCH) $(INCLUDES) $< -o $@

mincomp: mincomp.cu
	nvcc -O3 -arch $(CUDA_ARCH) $(INCLUDES) $< -o $@

clean: 
	rm -f la mincomp
