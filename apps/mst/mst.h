/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#pragma once

#include "component.h"
#include "kernels/reduce.cuh"

struct comp_data {
  Shared<int> weight;
  Shared<int> edge;
  Shared<int> node;
  Shared<int> level;
  Shared<int> dstcomp;

  int *lvl;
  int *minwt;
  int *minedge; // absolute
  int *minnode;
  int *mindstcomp;
};

static void dump_comp_data(struct comp_data comp, int n, int lvl);

static void dump_comp_data(struct comp_data comp, int n, int lvl) {
  int *level, *minwt, *minedge, *minnode, *mindstcomp;

  level = comp.level.cpu_rd_ptr();
  minwt = comp.weight.cpu_rd_ptr();
  minedge = comp.edge.cpu_rd_ptr();
  minnode = comp.node.cpu_rd_ptr();
  mindstcomp = comp.dstcomp.cpu_rd_ptr();

  for(int i = 0; i < n; i++) {
    if(level[i] == lvl) 
    {
      fprintf(stderr, "%d: (%d) node %d edge %d weight %d dstcomp %d\n", i, level[i], minnode[i], minedge[i], minwt[i], mindstcomp[i]);
    }
  }

  comp.level.gpu_wr_ptr();
  comp.weight.gpu_wr_ptr();
  comp.edge.gpu_wr_ptr();
  comp.node.gpu_wr_ptr();
  comp.dstcomp.gpu_wr_ptr();
}
