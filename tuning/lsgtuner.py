#!/usr/bin/env python

# requires Python 2.7

import sys
import argparse
import opentuner
import os
import tuning_params

from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import ConfigurationManipulator
from opentuner.search.manipulator import IntegerParameter
from opentuner.search.manipulator import EnumParameter
from opentuner import Result
import opentuner.search.manipulator

log = opentuner.search.manipulator.log
PARAMS_FRAGMENT = open(os.path.join(os.path.dirname(__file__), "tuning_fragment.h")).read()

class IntegerStepParameter(IntegerParameter):
    def __init__(self, name, min_value, max_value, step_value, **kwargs):
        # adjust max_value so it is a valid value
        max_value_2 = ((max_value - min_value) / step_value) * step_value + min_value # floor

        if max_value_2 != max_value:
            log.warning("IntegerStepParameter '%s' max_value adjusted to valid range value %d from %d." % (name, max_value_2, max_value))
        
        super(IntegerStepParameter, self).__init__(name, min_value, max_value_2, **kwargs)

        self.step_value = step_value
        
        # for sanity checking only
        self.valid_values = range(self.min_value, self.max_value + 1, self.step_value)

    def _get_closest_value(self, value):
        assert value >= self.min_value and value <= self.max_value

        v = (value - self.min_value) % self.step_value
        if v == 0:
            assert value in self.valid_values
            return value

        flr = ((value - self.min_value) / self.step_value) * self.step_value + self.min_value
        ceil = ((value - self.min_value + self.step_value - 1) / self.step_value) * self.step_value + self.min_value

        # not needed, but I'm paranoid
        if flr < self.min_value: flr = self.min_value
        if ceil > self.max_value: ceil = self.max_value

        assert flr < ceil and flr >= self.min_value and ceil <= self.max_value, "flr: %d, ceil: %d" % (flr, ceil)

        if (value - flr) < (ceil - value):
            x = flr
        else:
            x = ceil
        
        assert x in self.valid_values, x

        return x

    def set_value(self, config, value):
        super(IntegerParameter, self).set_value(config, self._get_closest_value(value))

    def search_space_size(self):
        return int((self.max_value - self.min_value) / self.step_value + 1)

class LSGBinary(object):
    binary = ""
    inputs = []
    params = []
    custom_savers = {}
    runtime_re = None
    output_h = ""
    repeats = 3
    tuning_parameter_var = "TUNING_PARAMETERS"
    compile_fail_ok = False
    debug = False

    def get_make_variables(self, cfg):
        return ""

    def get_compile_cmdline(self, cfg):
        self.save_cfg(cfg)
        if hasattr(self, 'make_target'):
            return "make -B %s %s" % (self.make_target, self.get_make_variables(cfg))
        else:
            return "make -B %s %s" % (self.binary, self.get_make_variables(cfg))

    def get_run_cmdline(self):
        return ["%s %s" % (self.binary, i,) for i in self.inputs]

    def pre_compile_check_cfg(self, cfg):
        return True

    def save_cfg(self, cfg):
        if self.output_h:
            of = open(self.output_h, "w") # overwrites old data!
            ov = []
            print >>of, "#pragma once"
            print >>of, PARAMS_FRAGMENT
            for p in self.params:
                k = p.name
                v = cfg[k]
                
                if k in self.custom_savers:
                    print >>of, self.custom_savers[k](cfg, k, v)
                else:
                    print >>of, "#define %s %s" % (k, v)
                
                ov.append("%s %s\\n" % (k, v))

            print >>of, "static const char *%s = \"%s\";" % (self.tuning_parameter_var, "".join(ov))

            of.close()
        
    def save_final_config(self, cfg):
        self.save_cfg(cfg.data)

class GenericLSGTuner(MeasurementInterface):
    def __init__(self, *args, **kwargs):
        lsgbinary = kwargs['lsgbinary']
        del kwargs['lsgbinary']

        super(GenericLSGTuner, self).__init__(*args, **kwargs)
        self._lsgbinary = lsgbinary

        info = 'Autotuning for "%s" (%d.%d) RT=%d DRV=%d' % (tuning_params.GPU_NAME, 
                                                             tuning_params.GPU_VERSION_MAJOR,
                                                             tuning_params.GPU_VERSION_MINOR,
                                                             tuning_params.RT_VERSION,
                                                             tuning_params.DRV_VERSION)
        print >>sys.stderr, info
        log.info(info)

    def manipulator(self):
        mp = ConfigurationManipulator()

        for p in self._lsgbinary.params:
            mp.add_parameter(p)

        return mp

    def get_runtime(self, run_result):
        runtime = self._lsgbinary.runtime_re.search(run_result['stdout'])
        if runtime: return float(runtime.group(1))

        return None
        
    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data
        if not self._lsgbinary.pre_compile_check_cfg(cfg):
            return Result(state='ERROR', time=float('inf'))

        make_cmd = self._lsgbinary.get_compile_cmdline(cfg)

        print >>sys.stderr, make_cmd

        compile_result = self.call_program(make_cmd)
        if compile_result['returncode'] != 0:
            if self._lsgbinary.debug:
                print >>sys.stderr, compile_result['stderr']

            if self._lsgbinary.compile_fail_ok: 
                return Result(state='ERROR', time=float('inf'))
            else:
                sys.exit(1)
        
        rc = self._lsgbinary.get_run_cmdline()
        error = False
        rt = 0.0
        for rrc in rc:

            rtt = 0.0

            for i in range(0, self._lsgbinary.repeats):
                run_result = self.call_program(rrc)
                if run_result['returncode'] == 0:
                    if self._lsgbinary.runtime_re:
                        if self._lsgbinary.debug:
                            print >>sys.stderr, run_result['stdout']

                        runtime = self.get_runtime(run_result)
                        assert runtime is not None
                        rtt += runtime
                    else:
                        rtt += run_result['time']
                else:
                    print >>sys.stderr, run_result['stderr']

                    if i > 0:
                        log.warning('Multiple run has second run failing.')
                        print >>sys.stderr, cfg
                        #assert False

                    error = True
                    break
                
            if error:
                break
            else:
                rt += rtt / self._lsgbinary.repeats

        if error:
            return Result(state='ERROR', time=float('inf'))
        else:
            return Result(time=rt)

    def save_final_config(self, configuration):
        """
        called at the end of autotuning with the best resultsdb.models.Configuration
        """
        print "Final configuration", configuration.data
        self._lsgbinary.save_final_config(configuration)

# if __name__ == "__main__": 
#     argparser = opentuner.default_argparser()
#     BFSTuner.main(argparser.parse_args())

