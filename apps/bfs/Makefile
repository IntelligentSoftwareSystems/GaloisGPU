# expect to be in ROOT/gensrc/app/

INCLUDES=-I../../rt/include -I../../rt/include/mgpu/include
LIBDIR=../../rt/lib
LIBS=-lggrt -lcurand
SKELAPP=../../skelapp
ARCH=sm_35
EXTRAFLAGS=-g -O3 -DCSRG_TEX
COMPRESS_LIBS=-lz
SNAPPY_LIBDIR=~/.local/lib

ifeq ($(USE_SNAPPY),1)
  COMPRESS_LIBS+=-lsnappy -L $(SNAPPY_LIBDIR)
endif

ARTIFACT=1
ifeq ($(ARTIFACT),1)
	NVCCFLAGS+=-D_FORCE_INLINES -L /usr/lib/x86_64-linux-gnu/ -Xptxas -v
endif

all: bfs

ifneq ($(wildcard kernel-nontex.cu),)
all: bfs_nontex
endif

$(SKELAPP)/mgpuhost.o: ../../rt/include/mgpu/src/mgpuhost.cu
	nvcc -dc -arch $(ARCH) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< -o $@

$(SKELAPP)/mgpucontext.o: ../../rt/include/mgpu/src/mgpucontext.cu
	nvcc -dc -arch $(ARCH) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< -o $@

$(SKELAPP)/mgpuutil.o: ../../rt/include/mgpu/src/mgpuutil.cpp
	nvcc -dc -arch $(ARCH) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< -o $@

$(SKELAPP)/skel.o: $(SKELAPP)/skel.cu
	nvcc -dc -arch $(ARCH) $(CXXFLAGS) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< -o $@

$(SKELAPP)/skel-nontex.o: $(SKELAPP)/skel.cu
	nvcc -dc -arch $(ARCH) $(CXXFLAGS) -g -O3 $(NVCCFLAGS) $(INCLUDES) $< -o $@

%.o: %.cu
#	nvcc -lineinfo -dc -arch $(ARCH) $(CXXFLAGS) $(EXTRAFLAGS) $(INCLUDES) $< $(EXTRA_SRC) -o $@ 
	nvcc -dc -arch $(ARCH) $(CXXFLAGS) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< $(EXTRA_SRC) -o $@ 

# this is only needed if the AST contained 'CSRGraphTy' instead of Graph().param() or such.
kernel-nontex.o: kernel-nontex.cu
	nvcc -dc -arch $(ARCH) $(CXXFLAGS) -g -O3 $(INCLUDES) $< $(EXTRA_SRC) -o $@ 

support-nontex.o: support.cu
	nvcc -dc -arch $(ARCH) $(CXXFLAGS) -g -O3 $(INCLUDES) $< $(EXTRA_SRC) -o $@ 

%.cubin: %.cu
	nvcc -cubin -dc -arch $(ARCH) $(CXXFLAGS) $(EXTRAFLAGS) $(NVCCFLAGS) $(INCLUDES) $< $(EXTRA_SRC) -o $@ 


bfs: $(SKELAPP)/skel.o kernel.o support.o $(SKELAPP)/mgpucontext.o $(SKELAPP)/mgpuutil.o
	nvcc -g $(EXTRAFLAGS) -arch $(ARCH) $(CXXFLAGS) $(NVCCFLAGS) $(INCLUDES) -L$(LIBDIR)  -o $@ $^ $(LIBS) -lcudadevrt $(COMPRESS_LIBS)

bfs_nontex: $(SKELAPP)/skel-nontex.o kernel-nontex.o support-nontex.o $(SKELAPP)/mgpucontext.o $(SKELAPP)/mgpuutil.o
	nvcc -g -O3 -arch $(ARCH) $(CXXFLAGS) $(NVCCFLAGS) $(INCLUDES) -L$(LIBDIR) $(LIBS) -o $@ $^ -lcudadevrt $(COMPRESS_LIBS)

bfs-dp: $(SKELAPP)/skel.o kernel.o support.o
	nvcc $(EXTRAFLAGS) -g -arch $(ARCH) $(NVCCFLAGS) $(INCLUDES) -o $@ $^ -lcudadevrt -L$(LIBDIR) $(LIBS) 

.phony: backup compare clean

clean:
	rm -f $(SKELAPP)/skel.o kernel.o support.o $(SKELAPP)/mgpucontext.o $(SKELAPP)/mgpuutil.o

backup: kernel.cu.1

%.cu.1: %.cu
	cp $< $@

compare:
	diff -u kernel.cu.1 kernel.cu
