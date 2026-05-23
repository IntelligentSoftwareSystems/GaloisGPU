include ../../arch.mk
INCLUDES=-I../include -I../include/mgpu/include
LIBDIR=../lib
SNAPPY_INC=~/.local/include
SNAPPY_LIB=~/.local/lib

ARTIFACT=1
ifeq ($(ARTIFACT),1)
	NVCCFLAGS+=-D_FORCE_INLINES -L /usr/lib/x86_64-linux-gnu/
endif

ifeq ($(USE_SNAPPY), 1)
	SNAPPY=-I $(SNAPPY_INC) -L $(SNAPPY_LIB) -DUSE_SNAPPY
	SNAPPY_OBJS=snappy.o
else
	SNAPPY=
	SNAPPY_OBJS=
endif

libggrt: $(LIBDIR) $(LIBDIR)/libggrt.a

$(LIBDIR):
	mkdir -p $(LIBDIR)

$(LIBDIR)/libggrt.a: csr_graph.o csr_graph.link.o ggc_rt.o ggc_rt.link.o bmk2.o instr.o $(SNAPPY_OBJS)
	nvcc -lib $^ -o $@

csr_graph.o: csr_graph.cu ../include/csr_graph.h

ggc_rt.o: ggc_rt.cu ../include/ggc_rt.h

instr.o: instr.cu ../include/instr.h

snappy.o: ../include/snfile.h

%.o: %.cu
	nvcc -g -dc -O3 $(CUDA_ARCH) $(INCLUDES) $(NVCCFLAGS) $< -o $@ $(SNAPPY)

%.o: %.c
	gcc -O3 -g -c $(INCLUDES) $< -o $@ $(SNAPPY)

snappy_test: snappy_test.o snappy.o
	gcc -g $(INCLUDES) $^ -o $@ $(SNAPPY)

%.link.o: %.o %.cu
	nvcc -g -dlink -O3 $(CUDA_ARCH) $(INCLUDES) $(NVCCFLAGS) $< -o $@

clean: 
	rm -f $(LIBDIR)/libggrt.a
	rm -f csr_graph.o csr_graph.link.o
	rm -f ggc_rt.o ggc_rt.link.o
	rm -f bmk2.o
	rm -f instr.o
