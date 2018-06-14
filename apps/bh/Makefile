TOPLEVEL := ../..
APP=bh
SRC=main.cu
INC=bh_tuning.h

include ../common.mk

.phony: tuning

tuning:
	python ../../tuning/bhtuner.py --parallelism 1 --test-limit 100
