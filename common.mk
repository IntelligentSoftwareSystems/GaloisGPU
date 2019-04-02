BIN    	  := $(TOPLEVEL)/bin
INPUTS 	  := $(TOPLEVEL)/inputs
NVCC      := nvcc
GCC       := g++
CC        := $(GCC)
CUB_DIR   := $(TOPLEVEL)/cub-1.2.0
ifdef debug
FLAGS := $(CUDA_ARCH) -g -DLSGDEBUG=1 -G -Xptxas -v #-lineinfo
else
FLAGS := -O3 $(CUDA_ARCH) -w
endif
INCLUDES := -I $(TOPLEVEL)/include -I $(CUB_DIR)
LINKS := 
EXTRA := $(FLAGS) $(INCLUDES) $(LINKS)

.PHONY: clean support 

ifdef APP
$(APP): $(SRC) $(INC)
	$(NVCC) $(EXTRA) -o $@ $<
	cp $@ $(BIN)

support: $(SUPPORT)

clean: 
	rm -f $(APP) $(BIN)/$(APP)

endif
