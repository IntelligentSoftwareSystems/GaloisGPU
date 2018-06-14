BIN    		:= $(TOPLEVEL)/bin
INPUTS 		:= $(TOPLEVEL)/inputs

NVCC 		:= nvcc
GCC  		:= g++
CC := $(GCC)
CUB_DIR := $(TOPLEVEL)/cub-1.2.0

COMPUTECAPABILITY := sm_35
ifdef debug
FLAGS := -arch=$(COMPUTECAPABILITY) -g -DLSGDEBUG=1 -G
else
# including -lineinfo -G causes launches to fail because of lack of resources, pity.
FLAGS := -O3 -arch=$(COMPUTECAPABILITY) -g -Xptxas -v #-lineinfo -G
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
