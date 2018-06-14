#!/usr/bin/env python

import lsgtuner
import re
import opentuner

from lsgtuner import IntegerStepParameter

class pta(lsgtuner.LSGBinary):
    debug=True
    binary = "./run.sh"
    make_target = "pta"
    inputs = ['../../inputs/tshark'] #, '../../inputs/vim'] #, '../../inputs/pine']
    params = [IntegerStepParameter('DEF_THREADS_PER_BLOCK', 32, 1024, 32),
              IntegerStepParameter('UPDATE_THREADS_PER_BLOCK', 32, 1024, 32),
              IntegerStepParameter('HCD_THREADS_PER_BLOCK', 32, 512, 32),
              IntegerStepParameter('COPY_INV_THREADS_PER_BLOCK', 32, 512, 32),
              IntegerStepParameter('STORE_INV_THREADS_PER_BLOCK', 32, 512, 32),
              IntegerStepParameter('GEP_INV_THREADS_PER_BLOCK', 32, 512, 32),
              ]
    runtime_re = re.compile(r"SOLVE runtime2: ([0-9.]+) ms.")
    output_h = "pta_tuning.h"

if __name__ == "__main__": 
    argparser = opentuner.default_argparser()
    lsgtuner.GenericLSGTuner.main(argparser.parse_args(), lsgbinary=pta())

