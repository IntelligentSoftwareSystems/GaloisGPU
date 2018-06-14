#!/usr/bin/env python

import lsgtuner
import re
import opentuner

def blksize_output(cfg, k, v):
    return "#define DBLKSIZE %s" % (v,)

class bfswlc(lsgtuner.LSGBinary):
    binary = "./bfs-wlc"
    inputs = ['../../inputs/USA-road-d.NY.gr'] #, '../../inputs/r4-2e23.gr', '../../inputs/rmat22.gr']]
    params = [lsgtuner.IntegerStepParameter('BLKSIZE', 32, 704, 32)]
    runtime_re = re.compile(r"runtime \[worklistc\] = ([0-9.]+) ms.")
    output_h = "bfs_wlc_tuning.h"
    custom_savers = {'BLKSIZE': blksize_output}

    def get_make_variables(self, cfg):
        return "BLKSIZE=%d" % (cfg['BLKSIZE'],)

if __name__ == "__main__": 
    argparser = opentuner.default_argparser()
    lsgtuner.GenericLSGTuner.main(argparser.parse_args(), lsgbinary=bfswlc())

