# expect to be in ROOT/app/
include ../../arch.mk
INCLUDES=-I../../rt/include -I../../rt/include/mgpu/include
LIBDIR=../../rt/lib
LIBS=-lggrt -lcurand
SKELAPP=../../skelapp
EXTRAFLAGS=-g -O3 -w
COMPRESS_LIBS=-lz
SNAPPY_LIBDIR=~/.local/lib

ifeq ($(USE_SNAPPY),1)
  COMPRESS_LIBS+=-lsnappy -L $(SNAPPY_LIBDIR)
endif

#ARTIFACT=1
ifeq ($(ARTIFACT),1)
	NVCCFLAGS+=-D_FORCE_INLINES -L /usr/lib/x86_64-linux-gnu/ -Xptxas -v
endif

all: pr

$(SKELAPP)/mgpuhost.o: ../../rt/include/mgpu/src/mgpuhost.cu
	nvcc -dc $(CUDA_ARCH) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< -o $@

$(SKELAPP)/mgpucontext.o: ../../rt/include/mgpu/src/mgpucontext.cu
	nvcc -dc $(CUDA_ARCH) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< -o $@

$(SKELAPP)/mgpuutil.o: ../../rt/include/mgpu/src/mgpuutil.cpp
	nvcc -dc $(CUDA_ARCH) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< -o $@

skel.o: $(SKELAPP)/skel.cu
	nvcc -dc $(CUDA_ARCH) $(CXXFLAGS) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< -o $@

%.o: %.cu
#	nvcc -lineinfo -dc $(CUDA_ARCH) $(CXXFLAGS) $(EXTRAFLAGS) $(INCLUDES) $< $(EXTRA_SRC) -o $@ 
	nvcc -dc $(CUDA_ARCH) $(CXXFLAGS) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< $(EXTRA_SRC) -o $@ 

%.cubin: %.cu
	nvcc -cubin -dc $(CUDA_ARCH) $(CXXFLAGS) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< $(EXTRA_SRC) -o $@ 


pr: skel.o kernel.o support.o $(SKELAPP)/mgpucontext.o $(SKELAPP)/mgpuutil.o
	nvcc -g $(EXTRAFLAGS) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) $(INCLUDES) -L$(LIBDIR)  -o $@ $^ $(LIBS) -lcudadevrt $(COMPRESS_LIBS)
	cp $@ ../../bin

pr-dp: skel.o kernel.o support.o
	nvcc $(EXTRAFLAGS) -g $(CUDA_ARCH) $(NVCCFLAGS) $(INCLUDES) -o $@ $^ -lcudadevrt -L$(LIBDIR) $(LIBS) 

.phony: backup compare clean

clean:
	rm -f pr skel.o kernel.o support.o $(SKELAPP)/mgpucontext.o $(SKELAPP)/mgpuutil.o

backup: kernel.cu.1

%.cu.1: %.cu
	cp $< $@

compare:
	diff -u kernel.cu.1 kernel.cu
