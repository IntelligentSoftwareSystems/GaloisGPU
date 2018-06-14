#include <cuda.h>
#include "gg.h"

int main(int argc, char *argv[]) {
  CSRGraph g, gg;

  g.readFromGR(argv[1]);
  
  if(g.nnodes < 10) {
    for(int r = 0; r < g.nnodes; r++) {
      for(int c = 0; c < g.getOutDegree(r); c++) {
	printf("%d -> %d %d\n", r+1, g.getDestination(r, c)+1, g.getWeight(r, c));
      }
    }
  }

  g.copy_to_gpu(gg);
  
  g.dealloc();
  gg.dealloc();
}
