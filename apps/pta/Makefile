TOPLEVEL := ../..
APP=pta
SRC=main.cu
INC=andersen.h andersen.cu pta_tuning.h

include ../../arch.mk
include ../../common.mk

.phony: tuning

tuning: 
	python ../../tuning/ptatuner.py --parallelism 1 --test-limit 500 --no-dups
