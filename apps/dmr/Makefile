INCLUDES=-I../../rt/include -I../../rt/include/mgpu/include
LIBDIR=../../rt/lib
LIBS=-lggrt -lz
SKELAPP=../../skelapp
SNAPPY_LIBDIR=~/.local/lib
ARCH=sm_35

ifeq ($(USE_SNAPPY),1)
  LIBS+=-lsnappy -L $(SNAPPY_LIBDIR)
endif

ARTIFACT=1
ifeq ($(ARTIFACT),1)
	NVCCFLAGS+=-D_FORCE_INLINES
endif

dmr: dmr.cu devel.h geomprim.h dmr.h meshfiles.h dmrggc.inc main.inc
	nvcc -O3 -g -arch $(ARCH) $(NVCCFLAGS) $(INCLUDES) -L$(LIBDIR) -o $@ $<  $(LIBS)

dmr_nontex: dmr-nontex.cu devel.h geomprim.h dmr.h meshfiles.h dmrggc.inc main.inc
	nvcc -O3 -g -arch $(ARCH) $(NVCCFLAGS) $(INCLUDES) -L$(LIBDIR) -o $@ $<  $(LIBS)

backup: dmr.cu.1

%.cu.1: %.cu
	cp $< $@

compare:
	diff -u dmr.cu.1 dmr.cu
