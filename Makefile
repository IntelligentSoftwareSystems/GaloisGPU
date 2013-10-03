TOPLEVEL := .
APPS     := $(wildcard $(TOPLEVEL)/apps/*/.)
TOOLS    := $(wildcard $(TOPLEVEL)/tools/*/.)

#all: 
#	$(MAKE) -f apps/common.mk

include apps/common.mk
