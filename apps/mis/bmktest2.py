import bmk2
from irglprops import irgl_bmk, PERF_RE, get_mis_checker

class mis_irgl(irgl_bmk):
	bmk = 'mis'
	variant = 'irgl'

	def filter_inputs(self, inputs):
		return filter(lambda x: x.props.format == 'bin/galois' and "symmetric" in x.props.flags, inputs)

	def get_run_spec(self, bmkinput):
		x = bmk2.RunSpec(self, bmkinput)
		x.set_binary(self.props._cwd, 'test')
		x.set_arg(bmkinput.props.file, bmk2.AT_INPUT_FILE)
		x.set_arg('-o')
		x.set_arg('@output', bmk2.AT_TEMPORARY_OUTPUT)
		x.set_checker(bmk2.ExternalChecker(get_mis_checker(bmkinput.props.file)))
		x.set_perf(bmk2.PerfRE(PERF_RE))
		return x

#mis rmat12.sym,USA-road-d.USA.sym,sample.sym,USA-road-d.NY.sym,2d-2e20.sym,USA-road-d.CAL.sym,USA-road-d.FLA.sym
BINARIES = [mis_irgl()]
