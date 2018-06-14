#!/usr/bin/env python

import lsgtuner
import re
import opentuner
import os
import tuning_params

class mst(lsgtuner.LSGBinary):
    binary = "./mst"
    inputs = ['../../inputs/USA-road-d.FLA.sym.gr', '../../inputs/rmat12.sym.gr']
    params = [lsgtuner.IntegerStepParameter('FINDCOMPMINTWO_BLKSIZE', 32, 384, 32),
              lsgtuner.IntegerStepParameter('KCONF_BLKSIZE', 32, tuning_params.MAX_TPB, 32)]

    runtime_re = re.compile(r"runtime \[mst\] = ([0-9.]+) ms.")
    output_h = "mst_tuning.h"
    tuning_parameter_var = "MST_TUNING_PARAMETERS"

if __name__ == "__main__": 
    argparser = opentuner.default_argparser()
    lsgtuner.GenericLSGTuner.main(argparser.parse_args(), lsgbinary=mst())

