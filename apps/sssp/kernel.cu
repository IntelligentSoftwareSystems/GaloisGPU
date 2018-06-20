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

#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraphTex &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=True $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=texture $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
extern int DELTA;
typedef int edge_data_type;
typedef int node_data_type;
typedef int * gint_p;
extern const node_data_type INF = INT_MAX;
static const int __tb_one = 1;
static const int __tb_gg_main_pipe_1_gpu_gb = 256;
static const int __tb_sssp_kernel = TB_SIZE;
static const int __tb_remove_dups = TB_SIZE;
__global__ void kernel(CSRGraphTex graph, int src)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type node_end;
  // FP: "1 -> 2;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    graph.node_data[node] = (node == src) ? 0 : INF ;
  }
  // FP: "4 -> 5;
}
__device__ void remove_dups_dev(int * marks, Worklist2 in_wl, Worklist2 out_wl, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type wlnode_end;
  index_type wlnode2_end;
  // FP: "1 -> 2;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode, node);
    marks[node] = wlnode;
  }
  // FP: "6 -> 7;
  gb.Sync();
  // FP: "7 -> 8;
  wlnode2_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode2 = 0 + tid; wlnode2 < wlnode2_end; wlnode2 += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode2, node);
    if (marks[node] == wlnode2)
    {
      index_type _start_25;
      _start_25 = (out_wl).setup_push_warp_one();;
      (out_wl).do_push(_start_25, 0, node);
    }
  }
  // FP: "16 -> 17;
}
__global__ void remove_dups(int * marks, Worklist2 in_wl, Worklist2 out_wl, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  // FP: "1 -> 2;
  remove_dups_dev(marks, in_wl, out_wl, gb);
  // FP: "2 -> 3;
}
__device__ void sssp_kernel_dev(CSRGraphTex graph, int delta, Worklist2 in_wl, Worklist2 out_wl, Worklist2 re_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_sssp_kernel;
  index_type wlnode_end;
  index_type wlnode_rup;
  // FP: "1 -> 2;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  // FP: "2 -> 3;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;
  // FP: "3 -> 4;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  // FP: "4 -> 5;
  __shared__ npsTy nps ;
  // FP: "5 -> 6;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  wlnode_rup = ((0) + roundup(((*((volatile index_type *) (in_wl).dindex)) - (0)), (blockDim.x)));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_rup; wlnode += nthreads)
  {
    int node;
    bool pop;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = (in_wl).pop_id(wlnode, node);
    pop = pop && !(graph.node_data[node] == INF);
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { int node; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].node = node;
    if (pop)
    {
      _np.size = (graph).getOutDegree(node);
      _np.start = (graph).getFirstEdge(node);
    }
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    if (threadIdx.x == 0)
    {
    }
    __syncthreads();
    {
      const int warpid = threadIdx.x / 32;
      const int _np_laneid = cub::LaneId();
      while (__any(_np.size >= _NP_CROSSOVER_WP))
      {
        if (_np.size >= _NP_CROSSOVER_WP)
        {
          nps.warp.owner[warpid] = _np_laneid;
        }
        if (nps.warp.owner[warpid] == _np_laneid)
        {
          nps.warp.start[warpid] = _np.start;
          nps.warp.size[warpid] = _np.size;
          nps.warp.src[warpid] = threadIdx.x;
          _np.start = 0;
          _np.size = 0;
        }
        index_type _np_w_start = nps.warp.start[warpid];
        index_type _np_w_size = nps.warp.size[warpid];
        assert(nps.warp.src[warpid] < __kernel_tb_size);
        node = _np_closure[nps.warp.src[warpid]].node;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type edge;
          edge = _np_w_start +_np_ii;
          {
            index_type dst = graph.getAbsDestination(edge);
            edge_data_type wt = graph.getAbsWeight(edge);
            if (graph.node_data[dst] > graph.node_data[node] + wt)
            {
              atomicMin(graph.node_data + dst, graph.node_data[node] + wt);
              if (graph.node_data[node] + wt <= delta)
              {
                index_type _start_66;
                _start_66 = (re_wl).setup_push_warp_one();;
                (re_wl).do_push(_start_66, 0, dst);
              }
              else
              {
                (out_wl).push(dst);
              }
            }
          }
        }
      }
      __syncthreads();
    }

    __syncthreads();
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    while (_np.work())
    {
      int _np_i =0;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      __syncthreads();

      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        node = _np_closure[nps.fg.src[_np_i]].node;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst = graph.getAbsDestination(edge);
          edge_data_type wt = graph.getAbsWeight(edge);
          if (graph.node_data[dst] > graph.node_data[node] + wt)
          {
            atomicMin(graph.node_data + dst, graph.node_data[node] + wt);
            if (graph.node_data[node] + wt <= delta)
            {
              index_type _start_66;
              _start_66 = (re_wl).setup_push_warp_one();;
              (re_wl).do_push(_start_66, 0, dst);
            }
            else
            {
              (out_wl).push(dst);
            }
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    node = _np_closure[threadIdx.x].node;
  }
}
__global__ void __launch_bounds__(TB_SIZE, 2) sssp_kernel(CSRGraphTex graph, int delta, Worklist2 in_wl, Worklist2 out_wl, Worklist2 re_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_sssp_kernel;
  if (tid == 0)
    in_wl.reset_next_slot();

  // FP: "1 -> 2;
  sssp_kernel_dev(graph, delta, in_wl, out_wl, re_wl);
  // FP: "2 -> 3;
}
void gg_main_pipe_1(CSRGraphTex& gg, gint_p glevel, int& curdelta, int& i, int DELTA, GlobalBarrier& remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  // FP: "1 -> 2;
  while (pipe.in_wl().nitems())
  {
    while (pipe.in_wl().nitems())
    {
      pipe.out_wl().will_write();
      pipe.re_wl().will_write();
      sssp_kernel <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
      pipe.in_wl().swap_slots();
      pipe.retry2();
    }
    pipe.advance2();
    pipe.out_wl().will_write();
    remove_dups <<<remove_dups_blocks, __tb_remove_dups>>>(glevel, pipe.in_wl(), pipe.out_wl(), remove_dups_barrier);
    pipe.in_wl().swap_slots();
    pipe.advance2();
    i++;
    curdelta += DELTA;
  }
  // FP: "6 -> 7;
}
__global__ void __launch_bounds__(__tb_gg_main_pipe_1_gpu_gb) gg_main_pipe_1_gpu_gb(CSRGraphTex gg, gint_p glevel, int curdelta, int i, int DELTA, GlobalBarrier remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2> pipe, int* cl_curdelta, int* cl_i, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_gg_main_pipe_1_gpu_gb;
  // FP: "1 -> 2;
  curdelta = *cl_curdelta;
  i = *cl_i;
  // FP: "2 -> 3;
  while (pipe.in_wl().nitems())
  {
    while (pipe.in_wl().nitems())
    {
      if (tid == 0)
        pipe.in_wl().reset_next_slot();
      sssp_kernel_dev (gg, curdelta, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
      pipe.in_wl().swap_slots();
      gb.Sync();
      pipe.retry2();
    }
    gb.Sync();
    pipe.advance2();
    if (tid == 0)
      pipe.in_wl().reset_next_slot();
    remove_dups_dev (glevel, pipe.in_wl(), pipe.out_wl(), gb);
    pipe.in_wl().swap_slots();
    gb.Sync();
    pipe.advance2();
    i++;
    curdelta += DELTA;
  }
  // FP: "7 -> 8;
  gb.Sync();
  // FP: "8 -> 9;
  if (tid == 0)
  {
    *cl_curdelta = curdelta;
    *cl_i = i;
  }
  // FP: "11 -> 12;
}
__global__ void gg_main_pipe_1_gpu(CSRGraphTex gg, gint_p glevel, int curdelta, int i, int DELTA, GlobalBarrier remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2> pipe, dim3 blocks, dim3 threads, int* cl_curdelta, int* cl_i)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_one;
  // FP: "1 -> 2;
  curdelta = *cl_curdelta;
  i = *cl_i;
  // FP: "2 -> 3;
  while (pipe.in_wl().nitems())
  {
    while (pipe.in_wl().nitems())
    {
      sssp_kernel <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
      pipe.in_wl().swap_slots();
      cudaDeviceSynchronize();
      pipe.retry2();
    }
    cudaDeviceSynchronize();
    pipe.advance2();
    remove_dups <<<remove_dups_blocks, __tb_remove_dups>>>(glevel, pipe.in_wl(), pipe.out_wl(), remove_dups_barrier);
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    i++;
    curdelta += DELTA;
  }
  // FP: "7 -> 8;
  if (tid == 0)
  {
    *cl_curdelta = curdelta;
    *cl_i = i;
  }
  // FP: "10 -> 11;
}
void gg_main_pipe_1_wrapper(CSRGraphTex& gg, gint_p glevel, int& curdelta, int& i, int DELTA, GlobalBarrier& remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  static GlobalBarrierLifetime gg_main_pipe_1_gpu_gb_barrier;
  static bool gg_main_pipe_1_gpu_gb_barrier_inited;
  // FP: "1 -> 2;
  static const size_t gg_main_pipe_1_gpu_gb_residency = maximum_residency(gg_main_pipe_1_gpu_gb, __tb_gg_main_pipe_1_gpu_gb, 0);
  static const size_t gg_main_pipe_1_gpu_gb_blocks = GG_MIN(blocks.x, ggc_get_nSM() * gg_main_pipe_1_gpu_gb_residency);
  if(!gg_main_pipe_1_gpu_gb_barrier_inited) { gg_main_pipe_1_gpu_gb_barrier.Setup(gg_main_pipe_1_gpu_gb_blocks); gg_main_pipe_1_gpu_gb_barrier_inited = true;};
  // FP: "2 -> 3;
  if (false)
  {
    gg_main_pipe_1(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads);
  }
  else
  {
    int* cl_curdelta;
    int* cl_i;
    check_cuda(cudaMalloc(&cl_curdelta, sizeof(int) * 1));
    check_cuda(cudaMalloc(&cl_i, sizeof(int) * 1));
    check_cuda(cudaMemcpy(cl_curdelta, &curdelta, sizeof(int) * 1, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(cl_i, &i, sizeof(int) * 1, cudaMemcpyHostToDevice));

    // gg_main_pipe_1_gpu<<<1,1>>>(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads,cl_curdelta,cl_i);
    gg_main_pipe_1_gpu_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,cl_curdelta,cl_i, gg_main_pipe_1_gpu_gb_barrier);
    check_cuda(cudaMemcpy(&curdelta, cl_curdelta, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&i, cl_i, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(cl_curdelta));
    check_cuda(cudaFree(cl_i));
  }
  // FP: "5 -> 6;
}
void gg_main(CSRGraphTex& hg, CSRGraphTex& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  static GlobalBarrierLifetime remove_dups_barrier;
  static bool remove_dups_barrier_inited;
  gint_p glevel;
  PipeContextT<Worklist2> pipe;
  // FP: "1 -> 2;
  Shared<int> level (hg.nnodes);
  // FP: "2 -> 3;
  level.cpu_wr_ptr();
  // FP: "3 -> 4;
  static const size_t remove_dups_residency = maximum_residency(remove_dups, __tb_remove_dups, 0);
  static const size_t remove_dups_blocks = GG_MIN(blocks.x, ggc_get_nSM() * remove_dups_residency);
  if(!remove_dups_barrier_inited) { remove_dups_barrier.Setup(remove_dups_blocks); remove_dups_barrier_inited = true;};
  // FP: "4 -> 5;
  // FP: "5 -> 6;
  kernel <<<blocks, threads>>>(gg, 0);
  // FP: "6 -> 7;
  int i = 0;
  int curdelta = 0;
  // FP: "7 -> 8;
  printf("delta: %d\n", DELTA);
  // FP: "8 -> 9;
  glevel = level.gpu_wr_ptr();
  // FP: "9 -> 10;
  pipe = PipeContextT<Worklist2>(gg.nedges*2);
  pipe.in_wl().wl[0] = 0;
  pipe.in_wl().update_gpu(1);
  gg_main_pipe_1_wrapper(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads);
  pipe.free();
  // FP: "14 -> 15;
  printf("iterations: %d\n", i);
  // FP: "15 -> 16;
}
