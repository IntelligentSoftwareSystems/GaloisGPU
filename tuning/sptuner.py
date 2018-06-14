#!/usr/bin/env python

import lsgtuner
import re
import opentuner

from lsgtuner import IntegerStepParameter

class newsp(lsgtuner.LSGBinary):
    binary = "./nsp"
    inputs = ['../../inputs/random-16800-4000-3.cnf 3', '../../inputs/random-42000-10000-3.cnf 3', '../../inputs/random-42000-10000-5.cnf 5']
    params = [IntegerStepParameter('CALC_PI_VALUES_BLKSIZE', 32, 384, 32),
              IntegerStepParameter('UPDATE_ETA_BLKSIZE', 32, 384, 32),
              IntegerStepParameter('UPDATE_BIAS_BLKSIZE', 32, 384, 32),
              IntegerStepParameter('DECIMATE_2_BLKSIZE', 32, 384, 32),
              ]
    runtime_re = re.compile(r"runtime \[nsp\] = ([0-9.]+) ms.")
    output_h = "newsp_tuning.h"

if __name__ == "__main__": 
    argparser = opentuner.default_argparser()
    lsgtuner.GenericLSGTuner.main(argparser.parse_args(), lsgbinary=newsp())

