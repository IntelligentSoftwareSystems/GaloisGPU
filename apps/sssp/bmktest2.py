import bmk2
from irglprops import irgl_bmk, PERF_RE

class sssp_nf(irgl_bmk):
	bmk = 'sssp'
	variant = 'nf'

	def filter_inputs(self, inputs):
		return filter(lambda x: x.props.format == 'bin/galois' and len(x.props.flags) == 0, inputs)

	def get_run_spec(self, bmkinput):
		x = bmk2.RunSpec(self, bmkinput)
		x.set_binary(self.props._cwd, 'test')
		x.set_arg(bmkinput.props.file, bmk2.AT_INPUT_FILE)
		x.set_arg('-d')
		x.set_arg(bmkinput.props.nf_delta)
		x.set_arg('-o')
		x.set_arg('@output', bmk2.AT_TEMPORARY_OUTPUT)
		x.set_checker(bmk2.DiffChecker('@output', bmkinput.props.sssp_output))
		x.set_perf(bmk2.PerfRE(PERF_RE))
		return x

#sssp USA-road-d.NY,r4-2e23,rmat22,USA-road-d.USA,USA-road-d.CAL
BINARIES = [sssp_nf()]
