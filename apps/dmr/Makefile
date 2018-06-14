TOPLEVEL := ../..
APP=dmr
SRC=dmr.cu
INC=meshfiles.h geomprim.h dmr.h devel.h dmr_tuning.h

include ../common.mk

.phony: tuning

tuning:
	python ../../tuning/dmrtuner.py --parallelism 1 --test-limit 500 --no-dups
