#!/usr/bin/env python

import lsgtuner
import re
import opentuner
import os

class dmr(lsgtuner.LSGBinary):
    binary = "./dmr"
    inputs = ['../../inputs/250k.2 20']
    params = [lsgtuner.IntegerStepParameter('CHECK_TRIANGLES_BLKSIZE', 32, 512, 32),
              lsgtuner.IntegerStepParameter('REFINE_BLKSIZE', 32, 512, 32),
              ]
    runtime_re = re.compile(r"runtime \[dmr\] = ([0-9.]+) ms")
    output_h = "dmr_tuning.h"

if __name__ == "__main__": 
    argparser = opentuner.default_argparser()
    os.environ['SKIP_OUTPUT'] = '1'
    lsgtuner.GenericLSGTuner.main(argparser.parse_args(), lsgbinary=dmr())

