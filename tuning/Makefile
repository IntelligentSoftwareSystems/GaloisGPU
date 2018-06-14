all: th

th: tune_helper.cu
	nvcc $< -o $@

.phony: setup

setup: th
	./th > tuning_fragment.h
	./th python > tuning_params.py
