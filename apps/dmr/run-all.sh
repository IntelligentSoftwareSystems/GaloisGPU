#!/bin/bash

INPUTS="../../inputs/250k.2 ../../inputs/r1M ../../inputs/r5M"
ADDL_INPUTS=(20 20 12)
VARIANTS="dmr"
RUNS=3

. ../scripts/run-all-template.sh

