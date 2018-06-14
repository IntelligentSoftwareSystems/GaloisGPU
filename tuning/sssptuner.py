#!/usr/bin/env python

import lsgtuner
import re
import opentuner

class ssspwlc(lsgtuner.LSGBinary):
    binary = "./sssp-wlc"
    inputs = ['../../inputs/USA-road-d.NY.gr'] #, '../../inputs/r4-2e23.gr', '../../inputs/rmat22.gr']]
    params = [lsgtuner.IntegerStepParameter('BLKSIZE', 32, 512, 32)]
    runtime_re = re.compile(r"runtime \[worklistc\] = ([0-9.]+) ms.")
    output_h = "sssp_wlc_tuning.h"

if __name__ == "__main__": 
    argparser = opentuner.default_argparser()
    lsgtuner.GenericLSGTuner.main(argparser.parse_args(), lsgbinary=ssspwlc())

