include ../../arch.mk
INCLUDES=-I../../include

la: la.cu ../../include/lockarray.h
	nvcc -O3 -arch $(CUDA_ARCH) $(INCLUDES) $< -o $@

mincomp: mincomp.cu
	nvcc -O3 -arch $(CUDA_ARCH) $(INCLUDES) $< -o $@

clean: 
	rm -f la mincomp
