TOPLEVEL := .
IRAPPS := bfs mst sssp sgd
APPS := bh dmr pta
INPUT_URL := http://iss.ices.utexas.edu/projects/galois/downloads/lonestargpu2-inputs.tar.bz2
BIP_INPUT_URL := http://iss.ices.utexas.edu/projects/galois/downloads/lonestargpu21-bipartite-inputs.tar.xz
INPUT := lonestargpu2-inputs.tar.bz2
BIP_INPUT := lonestargpu21-bipartite-inputs.tar.xz

.PHONY: all clean inputs

all: $(APPS)
	for IRAPPS in $(IRAPPS); do make -C apps/$$IRAPPS; done

$(APPS):
	make -C apps/$@

$(IRAPPS):
	make -C apps/$@
include apps/common.mk

inputs:
	@echo "Downloading inputs ..."
	@wget $(INPUT_URL) -O $(INPUT)
	@wget $(BIP_INPUT_URL) -O $(BIP_INPUT)
	@echo "Uncompressing inputs ..."
	@tar xvf $(INPUT)
	@tar xvf $(BIP_INPUT)
	@rm $(INPUT) $(BIP_INPUT)
	@echo "Inputs available at $(TOPLEVEL)/inputs/"

clean:
	for APP in $(APPS); do make -C apps/$$APP clean; done 
	for IRAPPS in $(IRAPPS); do make -C apps/$$IRAPPS clean; done

