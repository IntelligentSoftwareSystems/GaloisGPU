import bmk2
from irglprops import irgl_bmk, PERF_RE

class mst_wl_test(irgl_bmk):
	bmk = 'mst'
	variant = 'wl-test'

	def filter_inputs(self, inputs):
		return filter(lambda x: x.props.format == 'bin/galois' and 'symmetric' in x.props.flags, inputs)

	def get_run_spec(self, bmkinput):
		x = bmk2.RunSpec(self, bmkinput)
		x.set_binary(self.props._cwd, 'test')
		x.set_arg(bmkinput.props.file, bmk2.AT_INPUT_FILE)
		x.set_checker(bmk2.REChecker('^final mstwt: %s$' % (bmkinput.props.mst_weight)))
		x.set_perf(bmk2.PerfRE(PERF_RE))
		return x

#mst rmat12.sym,USA-road-d.USA.sym,USA-road-d.NY.sym,2d-2e20.sym,USA-road-d.CAL.sym,USA-road-d.FLA.sym
BINARIES = [mst_wl_test()]
