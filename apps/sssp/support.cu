/* -*- mode: C++ -*- */

#include "gg.h"
#include <cassert>

const char *prog_opts = "d:";
const char *prog_usage = "[-d delta]";
const char *prog_args_usage = "";

int DELTA = 10000;
extern const int INF;

int process_prog_arg(int argc, char *argv[], int arg_start) {
   return 1;
}

void process_prog_opt(char c, char *optarg) {
  if(c == 'd') {
    DELTA = atoi(optarg);
    assert(DELTA > 0);
  }
}


void output(CSRGraphTy &g, const char *output_file) {
  FILE *f;

  if(!output_file)
    return;

  if(strcmp(output_file, "-") == 0)
    f = stdout;
  else
    f = fopen(output_file, "w");
    
  for(int i = 0; i < g.nnodes; i++) {
    if(g.node_data[i] == INF) {
      check_fprintf(f, "%d INF\n", i);
    } else {
      check_fprintf(f, "%d %d\n", i, g.node_data[i]);
    }
  }

}
