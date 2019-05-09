include ../../arch.mk
#CUDA_COMPILE_OPTIONS=-m64 -lineinfo $(CUDA_ARCH) -O3
CUDA_COMPILE_OPTIONS=$(CUDA_ARCH) -O3
INCLUDE_DIR=-I../../include

all: sgd

clean: 
	rm -f sgd

sgd: TestCudaSGD.cu SGDAsyncEdgeCu.h SGDCommonCu.h
	nvcc ${CUDA_COMPILE_OPTIONS} -DBLOCKSIZE=1 -DR=2 -DC=2 -I./block_diagonal $(INCLUDE_DIR) TestCudaSGD.cu -o sgd
	cp $@ ../../bin
