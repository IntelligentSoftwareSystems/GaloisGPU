#!/usr/bin/env python

import lsgtuner
import re
import opentuner
from tuning_params import *

from lsgtuner import IntegerStepParameter, EnumParameter, IntegerParameter

class bh(lsgtuner.LSGBinary):
    debug=True
    binary = "./bh"
    inputs = ['300000 10 0']
    repeats = 1
    compile_fail_ok = True
    params = [EnumParameter('THREADS1', [32, 64, 128, 256, 512, 1024]),  # must be a power of 2 
              IntegerStepParameter('THREADS2', 32, 512, 32),  #512
              IntegerStepParameter('THREADS3', 32, 512, 32),  #128
              IntegerStepParameter('THREADS4', 32, 512, 32), #64
              IntegerStepParameter('THREADS5', 32, 512, 32), #256
              IntegerStepParameter('THREADS6', 32, 1024, 32), #1024
              IntegerParameter('FACTOR1', 1, MAX_TB_PER_SM), # 3
              IntegerParameter('FACTOR2', 1, MAX_TB_PER_SM), # 3
              IntegerParameter('FACTOR3', 1, MAX_TB_PER_SM), # 6  /* must all be resident at the same time */
              IntegerParameter('FACTOR4', 1, MAX_TB_PER_SM), # 6  /* must all be resident at the same time */
              IntegerParameter('FACTOR5', 1, MAX_TB_PER_SM), # 5
              IntegerParameter('FACTOR6', 1, MAX_TB_PER_SM), # 1              
              ]

    runtime_re = re.compile(r"runtime: ([0-9.]+) s.")
    output_h = "bh_tuning.h"

    def pre_compile_check_cfg(self, cfg):
        for i in range(1,7):
            t = cfg["THREADS%d" % (i,)]
            f = cfg["FACTOR%d" % (i,)]

            if t * f > MAX_THREADS_PER_SM: return False

        return True

    def get_runtime(self, run_result):
        out = run_result['output']

        a = self.runtime_re.findall(out)
        if not a: return None

        t = sum([float(aa) for aa in a]) / len(a)
        return t


if __name__ == "__main__": 
    argparser = opentuner.default_argparser()
    lsgtuner.GenericLSGTuner.main(argparser.parse_args(), lsgbinary=bh())

