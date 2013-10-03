BIN    		:= $(TOPLEVEL)/bin
INPUTS 		:= $(TOPLEVEL)/inputs

NVCC 		:= nvcc
GCC  		:= g++

COMPUTECAPABILITY := sm_20
FLAGS := -O3 -arch=$(COMPUTECAPABILITY)
INCLUDES := -I $(TOPLEVEL)/include
LINKS := 

EXTRA := $(FLAGS) $(INCLUDES) $(LINKS)

.PHONY: all $(APPS) inputs

all: $(APPS)

$(APPS):
	$(NVCC) $(EXTRA) -o $(BIN)/$(notdir $(subst /.,,$@)) $@/main.cu
#$(MAKE) -C $@ all

inputs:
	@echo "Downloading inputs, this may take a minute..."
	@wget http://iss.ices.utexas.edu/projects/galois/downloads/lonestargpu-inputs.tar.gz -O $(TOPLEVEL)/inputs.tar.gz 2>/dev/null
	@echo "Uncompressing inputs, this may take another minute..."
	@tar xvzf $(TOPLEVEL)/inputs.tar.gz
	@rm $(TOPLEVEL)/inputs.tar.gz
	@echo "Inputs available at $(TOPLEVEL)/inputs/"

clean:
	rm $(BIN)/*
