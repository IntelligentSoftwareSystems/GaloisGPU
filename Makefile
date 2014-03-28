TOPLEVEL := .
APPS := bfs bh dmr mst pta sp sssp

.PHONY: all clean inputs

all: $(APPS)

$(APPS):
	make -C apps/$@

include apps/common.mk

inputs:
	@echo "Downloading inputs, this may take a minute..."
	@wget http://iss.ices.utexas.edu/projects/galois/downloads/lonestargpu-inputs.tar.gz -O $(TOPLEVEL)/inputs.tar.gz 2>/dev/null
	@echo "Uncompressing inputs, this may take another minute..."
	@tar xvzf $(TOPLEVEL)/inputs.tar.gz
	@rm $(TOPLEVEL)/inputs.tar.gz
	@echo "Inputs available at $(TOPLEVEL)/inputs/"

clean:
	for APP in $(APPS); do make -C apps/$$APP clean; done
#	rm -f $(BIN)/*
