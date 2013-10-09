BIN    		:= $(TOPLEVEL)/bin
INPUTS 		:= $(TOPLEVEL)/inputs

NVCC 		:= nvcc
GCC  		:= g++

COMPUTECAPABILITY := sm_20
FLAGS := -O3 -arch=$(COMPUTECAPABILITY)
INCLUDES := -I $(TOPLEVEL)/include
LINKS := 

EXTRA := $(FLAGS) $(INCLUDES) $(LINKS)

.PHONY: clean

ifdef APP
$(APP): $(SRC)
	$(NVCC) $(EXTRA) -o $(APP) $<
	cp $(APP) $(BIN)

clean: 
	rm -f $(APP) $(BIN)/$(APP)
endif