/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=True $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set(['lighten-wl']) $ np_factor=8 $ instrument=set([]) $ unroll=['pipes'] $ instrument_mode=None $ read_props=bfs-hybrid.props $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
typedef int edge_data_type;
typedef int node_data_type;
#define RESIDENCY 6
#define TB_SIZE_CX 128
#define WORKLISTSIZE 14336*2
extern const node_data_type INF = INT_MAX;
static const int __tb_one = 1;
static const int __tb_gg_main_pipe_1_gpu_gb = 128;
static const int __tb_contract = TB_SIZE;
static const int __tb_filter = TB_SIZE;
static const int __tb_contract_expand = TB_SIZE_CX;
static const int __tb_expand = TB_SIZE;
__global__ void kernel(CSRGraph graph, int src)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type node_end;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    graph.node_data[node] = INF ;
  }
  // FP: "5 -> 6;
  // FP: "4 -> 5;
  // FP: "6 -> 7;
}
__device__ void contract_expand_dev_light(CSRGraph graph, int LEVEL, Worklist2Light in_wl, Worklist2Light out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_contract_expand;
  index_type wlnode_end;
  index_type wlnode_rup;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  // FP: "3 -> 4;
  // FP: "2 -> 3;
  // FP: "4 -> 5;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;
  // FP: "5 -> 6;
  // FP: "3 -> 4;
  // FP: "6 -> 7;

  typedef cub::BlockScan<index_type, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct empty_np, struct fg_np<ITSIZE> > npsTy;

  // FP: "7 -> 8;
  // FP: "4 -> 5;
  // FP: "8 -> 9;
  __shared__ npsTy nps ;
  // FP: "9 -> 10;
  // FP: "5 -> 6;
  // FP: "10 -> 11;
  const int WHTSIZE = 64;
  __shared__ volatile int whash[__kernel_tb_size/32][WHTSIZE] ;
  // FP: "11 -> 12;
  // FP: "6 -> 7;
  // FP: "12 -> 13;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  wlnode_rup = ((0) + roundup(((*((volatile index_type *) (in_wl).dindex)) - (0)), (blockDim.x)));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_rup; wlnode += nthreads)
  {
    int node;
    bool pop;
    index_type _start_43;
    index_type _np_mps;
    index_type _np_mps_total;
    __shared__ index_type _np_pc__start_43;
    // FP: "13 -> 14;
    // FP: "7 -> 8;
    // FP: "14 -> 15;
    // FP: "15 -> 16;
    // FP: "8 -> 9;
    // FP: "16 -> 17;
    pop = (in_wl).pop_id(wlnode, node);
    // FP: "17 -> 18;
    // FP: "9 -> 10;
    // FP: "18 -> 19;
    if (pop)
    {
      pop = false;
      if (graph.node_data[node] == INF)
      {
        graph.node_data[node] = LEVEL;
        pop = true;
      }
    }
    // FP: "24 -> 25;
    // FP: "15 -> 16;
    // FP: "25 -> 26;
    if (pop)
    {
      const int warpid = threadIdx.x / 32;
      const int htentry = node & (WHTSIZE - 1);
      whash[warpid][htentry] = node;
      if (whash[warpid][htentry] == node)
      {
        whash[warpid][htentry] = threadIdx.x;
        pop = whash[warpid][htentry] == threadIdx.x;
      }
    }
    // FP: "32 -> 33;
    // FP: "22 -> 23;
    // FP: "33 -> 34;
    __syncthreads();
    // FP: "34 -> 35;
    // FP: "23 -> 24;
    // FP: "35 -> 36;
    // FP: "26 -> 27;
    // FP: "36 -> 37;
    // FP: "37 -> 38;
    // FP: "27 -> 28;
    // FP: "38 -> 39;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "39 -> 40;
    // FP: "28 -> 29;
    // FP: "40 -> 41;
    if (pop)
    {
      _np.size = (graph).getOutDegree(node);
      _np.start = (graph).getFirstEdge(node);
    }
    // FP: "43 -> 44;
    // FP: "31 -> 32;
    // FP: "44 -> 45;
    // FP: "45 -> 46;
    // FP: "32 -> 33;
    // FP: "46 -> 47;
    _np_mps = _np.size;
    // FP: "47 -> 48;
    // FP: "33 -> 34;
    // FP: "48 -> 49;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "49 -> 50;
    // FP: "34 -> 35;
    // FP: "50 -> 51;
    // FP: "51 -> 52;
    // FP: "35 -> 36;
    // FP: "52 -> 53;
    if (threadIdx.x == 0)
    {
      _np_pc__start_43 = (out_wl).push_range(_np_mps_total);;
    }
    // FP: "55 -> 56;
    // FP: "38 -> 39;
    // FP: "56 -> 57;
    __syncthreads();
    // FP: "57 -> 58;
    // FP: "39 -> 40;
    // FP: "58 -> 59;
    _np.total = _np_mps_total;
    _np.offset = _np_mps;
    // FP: "59 -> 60;
    // FP: "40 -> 41;
    // FP: "60 -> 61;
    _start_43 = _np_pc__start_43;
    // FP: "61 -> 62;
    // FP: "41 -> 42;
    // FP: "62 -> 63;
    while (_np.work())
    {
      // FP: "63 -> 64;
      // FP: "42 -> 43;
      // FP: "64 -> 65;
      int _np_i =0;
      // FP: "65 -> 66;
      // FP: "43 -> 44;
      // FP: "66 -> 67;
      _np.inspect(nps.fg.itvalue, ITSIZE);
      // FP: "67 -> 68;
      // FP: "44 -> 45;
      // FP: "68 -> 69;
      __syncthreads();
      // FP: "69 -> 70;
      // FP: "45 -> 46;
      // FP: "70 -> 71;

      // FP: "71 -> 72;
      // FP: "46 -> 47;
      // FP: "72 -> 73;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type edge;
        index_type edge_pos  = _np_i;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(edge);
          (out_wl).do_push(_start_43, edge_pos, dst);
        }
      }
      // FP: "80 -> 81;
      // FP: "54 -> 55;
      // FP: "81 -> 82;
      _np.execute_round_done(ITSIZE);
      // FP: "82 -> 83;
      // FP: "55 -> 56;
      // FP: "83 -> 84;
      _start_43 = _start_43 + ITSIZE;;
      // FP: "84 -> 85;
      // FP: "56 -> 57;
      // FP: "85 -> 86;
      __syncthreads();
    }
  }
  // FP: "88 -> 89;
  // FP: "59 -> 60;
  // FP: "89 -> 90;
}
__device__ void contract_expand_dev(CSRGraph graph, int LEVEL, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_contract_expand;
  index_type wlnode_end;
  index_type wlnode_rup;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  // FP: "3 -> 4;
  // FP: "2 -> 3;
  // FP: "4 -> 5;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;
  // FP: "5 -> 6;
  // FP: "3 -> 4;
  // FP: "6 -> 7;

  typedef cub::BlockScan<index_type, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct empty_np, struct fg_np<ITSIZE> > npsTy;

  // FP: "7 -> 8;
  // FP: "4 -> 5;
  // FP: "8 -> 9;
  __shared__ npsTy nps ;
  // FP: "9 -> 10;
  // FP: "5 -> 6;
  // FP: "10 -> 11;
  const int WHTSIZE = 64;
  __shared__ volatile int whash[__kernel_tb_size/32][WHTSIZE] ;
  // FP: "11 -> 12;
  // FP: "6 -> 7;
  // FP: "12 -> 13;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  wlnode_rup = ((0) + roundup(((*((volatile index_type *) (in_wl).dindex)) - (0)), (blockDim.x)));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_rup; wlnode += nthreads)
  {
    int node;
    bool pop;
    index_type _start_43;
    index_type _np_mps;
    index_type _np_mps_total;
    __shared__ index_type _np_pc__start_43;
    // FP: "13 -> 14;
    // FP: "7 -> 8;
    // FP: "14 -> 15;
    // FP: "15 -> 16;
    // FP: "8 -> 9;
    // FP: "16 -> 17;
    pop = (in_wl).pop_id(wlnode, node);
    // FP: "17 -> 18;
    // FP: "9 -> 10;
    // FP: "18 -> 19;
    if (pop)
    {
      pop = false;
      if (graph.node_data[node] == INF)
      {
        graph.node_data[node] = LEVEL;
        pop = true;
      }
    }
    // FP: "24 -> 25;
    // FP: "15 -> 16;
    // FP: "25 -> 26;
    if (pop)
    {
      const int warpid = threadIdx.x / 32;
      const int htentry = node & (WHTSIZE - 1);
      whash[warpid][htentry] = node;
      if (whash[warpid][htentry] == node)
      {
        whash[warpid][htentry] = threadIdx.x;
        pop = whash[warpid][htentry] == threadIdx.x;
      }
    }
    // FP: "32 -> 33;
    // FP: "22 -> 23;
    // FP: "33 -> 34;
    __syncthreads();
    // FP: "34 -> 35;
    // FP: "23 -> 24;
    // FP: "35 -> 36;
    // FP: "26 -> 27;
    // FP: "36 -> 37;
    // FP: "37 -> 38;
    // FP: "27 -> 28;
    // FP: "38 -> 39;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "39 -> 40;
    // FP: "28 -> 29;
    // FP: "40 -> 41;
    if (pop)
    {
      _np.size = (graph).getOutDegree(node);
      _np.start = (graph).getFirstEdge(node);
    }
    // FP: "43 -> 44;
    // FP: "31 -> 32;
    // FP: "44 -> 45;
    // FP: "45 -> 46;
    // FP: "32 -> 33;
    // FP: "46 -> 47;
    _np_mps = _np.size;
    // FP: "47 -> 48;
    // FP: "33 -> 34;
    // FP: "48 -> 49;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "49 -> 50;
    // FP: "34 -> 35;
    // FP: "50 -> 51;
    // FP: "51 -> 52;
    // FP: "35 -> 36;
    // FP: "52 -> 53;
    if (threadIdx.x == 0)
    {
      _np_pc__start_43 = (out_wl).push_range(_np_mps_total);;
    }
    // FP: "55 -> 56;
    // FP: "38 -> 39;
    // FP: "56 -> 57;
    __syncthreads();
    // FP: "57 -> 58;
    // FP: "39 -> 40;
    // FP: "58 -> 59;
    _np.total = _np_mps_total;
    _np.offset = _np_mps;
    // FP: "59 -> 60;
    // FP: "40 -> 41;
    // FP: "60 -> 61;
    _start_43 = _np_pc__start_43;
    // FP: "61 -> 62;
    // FP: "41 -> 42;
    // FP: "62 -> 63;
    while (_np.work())
    {
      // FP: "63 -> 64;
      // FP: "42 -> 43;
      // FP: "64 -> 65;
      int _np_i =0;
      // FP: "65 -> 66;
      // FP: "43 -> 44;
      // FP: "66 -> 67;
      _np.inspect(nps.fg.itvalue, ITSIZE);
      // FP: "67 -> 68;
      // FP: "44 -> 45;
      // FP: "68 -> 69;
      __syncthreads();
      // FP: "69 -> 70;
      // FP: "45 -> 46;
      // FP: "70 -> 71;

      // FP: "71 -> 72;
      // FP: "46 -> 47;
      // FP: "72 -> 73;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type edge;
        index_type edge_pos  = _np_i;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(edge);
          (out_wl).do_push(_start_43, edge_pos, dst);
        }
      }
      // FP: "80 -> 81;
      // FP: "54 -> 55;
      // FP: "81 -> 82;
      _np.execute_round_done(ITSIZE);
      // FP: "82 -> 83;
      // FP: "55 -> 56;
      // FP: "83 -> 84;
      _start_43 = _start_43 + ITSIZE;;
      // FP: "84 -> 85;
      // FP: "56 -> 57;
      // FP: "85 -> 86;
      __syncthreads();
    }
  }
  // FP: "88 -> 89;
  // FP: "59 -> 60;
  // FP: "89 -> 90;
}
__global__ void __launch_bounds__(TB_SIZE_CX, RESIDENCY) contract_expand(CSRGraph graph, int LEVEL, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_contract_expand;
  if (tid == 0)
    in_wl.reset_next_slot();

  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  contract_expand_dev(graph, LEVEL, in_wl, out_wl);
  // FP: "3 -> 4;
  // FP: "2 -> 3;
  // FP: "4 -> 5;
}
__global__ void __launch_bounds__(TB_SIZE, 8) expand(CSRGraph graph, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_expand;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlnode_end;
  index_type wlnode_rup;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  // FP: "3 -> 4;
  // FP: "2 -> 3;
  // FP: "4 -> 5;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;
  // FP: "5 -> 6;
  // FP: "3 -> 4;
  // FP: "6 -> 7;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct tb_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  // FP: "7 -> 8;
  // FP: "4 -> 5;
  // FP: "8 -> 9;
  __shared__ npsTy nps ;
  // FP: "9 -> 10;
  // FP: "5 -> 6;
  // FP: "10 -> 11;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  wlnode_rup = ((0) + roundup(((*((volatile index_type *) (in_wl).dindex)) - (0)), (blockDim.x)));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_rup; wlnode += nthreads)
  {
    int node;
    bool pop;
    index_type _start_54;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    __shared__ index_type _np_pc__start_54;
    // FP: "11 -> 12;
    // FP: "6 -> 7;
    // FP: "12 -> 13;
    // FP: "13 -> 14;
    // FP: "7 -> 8;
    // FP: "14 -> 15;
    pop = (in_wl).pop_id(wlnode, node);
    // FP: "15 -> 16;
    // FP: "8 -> 9;
    // FP: "16 -> 17;
    // FP: "17 -> 18;
    // FP: "9 -> 10;
    // FP: "18 -> 19;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "19 -> 20;
    // FP: "10 -> 11;
    // FP: "20 -> 21;
    if (pop)
    {
      _np.size = (graph).getOutDegree(node);
      _np.start = (graph).getFirstEdge(node);
    }
    // FP: "23 -> 24;
    // FP: "13 -> 14;
    // FP: "24 -> 25;
    // FP: "25 -> 26;
    // FP: "14 -> 15;
    // FP: "26 -> 27;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "27 -> 28;
    // FP: "15 -> 16;
    // FP: "28 -> 29;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "29 -> 30;
    // FP: "16 -> 17;
    // FP: "30 -> 31;
    // FP: "31 -> 32;
    // FP: "17 -> 18;
    // FP: "32 -> 33;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
      _np_pc__start_54 = (out_wl).push_range(_np_mps_total.el[0] + _np_mps_total.el[1]);;
    }
    // FP: "36 -> 37;
    // FP: "21 -> 22;
    // FP: "37 -> 38;
    __syncthreads();
    // FP: "38 -> 39;
    // FP: "22 -> 23;
    // FP: "39 -> 40;
    while (true)
    {
      // FP: "40 -> 41;
      // FP: "23 -> 24;
      // FP: "41 -> 42;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "44 -> 45;
      // FP: "26 -> 27;
      // FP: "45 -> 46;
      __syncthreads();
      // FP: "46 -> 47;
      // FP: "27 -> 28;
      // FP: "47 -> 48;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "48 -> 49;
        // FP: "28 -> 29;
        // FP: "49 -> 50;
        __syncthreads();
        // FP: "50 -> 51;
        // FP: "29 -> 30;
        // FP: "51 -> 52;
        break;
      }
      // FP: "53 -> 54;
      // FP: "31 -> 32;
      // FP: "54 -> 55;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.offset = _np_mps.el[0];
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "58 -> 59;
      // FP: "35 -> 36;
      // FP: "59 -> 60;
      __syncthreads();
      // FP: "60 -> 61;
      // FP: "36 -> 37;
      // FP: "61 -> 62;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "62 -> 63;
      // FP: "37 -> 38;
      // FP: "63 -> 64;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "66 -> 67;
      // FP: "40 -> 41;
      // FP: "67 -> 68;
      _start_54 = _np_pc__start_54 + nps.tb.offset;
      // FP: "68 -> 69;
      // FP: "41 -> 42;
      // FP: "69 -> 70;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type edge;
        edge = ns +_np_j;
        index_type edge_pos = _np_j;
        {
          index_type dst;
          dst = graph.getAbsDestination(edge);
          (out_wl).do_push(_start_54, edge_pos, dst);
        }
      }
      // FP: "77 -> 78;
      // FP: "49 -> 50;
      // FP: "78 -> 79;
      __syncthreads();
    }
    // FP: "80 -> 81;
    // FP: "51 -> 52;
    // FP: "81 -> 82;

    // FP: "82 -> 83;
    // FP: "52 -> 53;
    // FP: "83 -> 84;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "84 -> 85;
      // FP: "53 -> 54;
      // FP: "85 -> 86;
      const int _np_laneid = cub::LaneId();
      // FP: "86 -> 87;
      // FP: "54 -> 55;
      // FP: "87 -> 88;
      while (__any(_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB))
      {
        if (_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB)
        {
          nps.warp.owner[warpid] = _np_laneid;
        }
        if (nps.warp.owner[warpid] == _np_laneid)
        {
          nps.warp.offset[warpid] = _np_mps.el[0];
          nps.warp.start[warpid] = _np.start;
          nps.warp.size[warpid] = _np.size;

          _np.start = 0;
          _np.size = 0;
        }
        index_type _np_w_start = nps.warp.start[warpid];
        index_type _np_w_size = nps.warp.size[warpid];
        _start_54 =  _np_pc__start_54 + nps.warp.offset[warpid];
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type edge;
          edge = _np_w_start +_np_ii;
          index_type edge_pos = _np_ii;
          {
            index_type dst;
            dst = graph.getAbsDestination(edge);
            (out_wl).do_push(_start_54, edge_pos, dst);
          }
        }
      }
      // FP: "106 -> 107;
      // FP: "73 -> 74;
      // FP: "107 -> 108;
      __syncthreads();
      // FP: "108 -> 109;
      // FP: "74 -> 75;
      // FP: "109 -> 110;
    }

    // FP: "110 -> 111;
    // FP: "75 -> 76;
    // FP: "111 -> 112;
    __syncthreads();
    // FP: "112 -> 113;
    // FP: "76 -> 77;
    // FP: "113 -> 114;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "114 -> 115;
    // FP: "77 -> 78;
    // FP: "115 -> 116;
    _start_54 = _np_pc__start_54 + _np_mps_total.el[0];
    // FP: "116 -> 117;
    // FP: "78 -> 79;
    // FP: "117 -> 118;
    while (_np.work())
    {
      // FP: "118 -> 119;
      // FP: "79 -> 80;
      // FP: "119 -> 120;
      int _np_i =0;
      // FP: "120 -> 121;
      // FP: "80 -> 81;
      // FP: "121 -> 122;
      _np.inspect(nps.fg.itvalue, ITSIZE);
      // FP: "122 -> 123;
      // FP: "81 -> 82;
      // FP: "123 -> 124;
      __syncthreads();
      // FP: "124 -> 125;
      // FP: "82 -> 83;
      // FP: "125 -> 126;

      // FP: "126 -> 127;
      // FP: "83 -> 84;
      // FP: "127 -> 128;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type edge;
        index_type edge_pos  = _np_i;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(edge);
          (out_wl).do_push(_start_54, edge_pos, dst);
        }
      }
      // FP: "135 -> 136;
      // FP: "91 -> 92;
      // FP: "136 -> 137;
      _np.execute_round_done(ITSIZE);
      // FP: "137 -> 138;
      // FP: "92 -> 93;
      // FP: "138 -> 139;
      _start_54 = _start_54 + ITSIZE;;
      // FP: "139 -> 140;
      // FP: "93 -> 94;
      // FP: "140 -> 141;
      __syncthreads();
    }
  }
  // FP: "143 -> 144;
  // FP: "96 -> 97;
  // FP: "144 -> 145;
}
__global__ void __launch_bounds__(TB_SIZE, 8) contract(CSRGraph graph, int LEVEL, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlnode_end;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode, node);
    if (pop)
    {
      if (graph.node_data[node] == INF)
      {
        index_type _start_66;
        graph.node_data[node] = LEVEL;
        _start_66 = (out_wl).setup_push_warp_one();;
        (out_wl).do_push(_start_66, 0, node);
      }
    }
  }
  // FP: "14 -> 15;
  // FP: "13 -> 14;
  // FP: "15 -> 16;
}
__global__ void __launch_bounds__(TB_SIZE, 8) filter(ApproxBitsetByte visited, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlnode_end;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode, node);
    if (pop)
    {
      if (!visited.is_set(node))
      {
        index_type _start_81;
        visited.set(node);
        _start_81 = (out_wl).setup_push_warp_one();;
        (out_wl).do_push(_start_81, 0, node);
      }
    }
  }
  // FP: "14 -> 15;
  // FP: "13 -> 14;
  // FP: "15 -> 16;
}
void gg_main_pipe_1(CSRGraph& gg, int& LEVEL, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  while (pipe.in_wl().nitems())
  {
    pipe.out_wl().will_write();
    contract_expand <<<blocks, __tb_contract_expand>>>(gg, LEVEL, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    pipe.advance2();
    LEVEL++;
    if (pipe.in_wl().nitems() > WORKLISTSIZE)
    {
      break;
    }
  }
  // FP: "9 -> 10;
  // FP: "8 -> 9;
  // FP: "10 -> 11;
}
__global__ void __launch_bounds__(__tb_gg_main_pipe_1_gpu_gb) gg_main_pipe_1_gpu_gb(CSRGraph gg, int LEVEL, PipeContextT<Worklist2> pipe, int* cl_LEVEL, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_gg_main_pipe_1_gpu_gb;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  LEVEL = *cl_LEVEL;
  // FP: "3 -> 4;
  // FP: "2 -> 3;
  // FP: "4 -> 5;
  PipeContextLight pipe_light (pipe);
  int _light_index = 0;
  // FP: "5 -> 6;
  // FP: "3 -> 4;
  // FP: "6 -> 7;
  while (pipe.in_wl().nitems())
  {
    if (tid == 0)
      pipe_light.wl[_light_index].reset_next_slot();
    contract_expand_dev_light (gg, LEVEL, pipe_light.wl[_light_index], pipe_light.wl[_light_index ^ 1]);
    pipe_light.wl[_light_index].swap_slots();
    gb.Sync();
    pipe.advance2();
    _light_index ^= 1;
    LEVEL++;
    if (pipe.in_wl().nitems() > WORKLISTSIZE)
    {
      break;
    }
    if (tid == 0)
      pipe_light.wl[_light_index].reset_next_slot();
    contract_expand_dev_light (gg, LEVEL, pipe_light.wl[_light_index], pipe_light.wl[_light_index ^ 1]);
    pipe_light.wl[_light_index].swap_slots();
    gb.Sync();
    pipe.advance2();
    _light_index ^= 1;
    LEVEL++;
    if (pipe.in_wl().nitems() > WORKLISTSIZE)
    {
      break;
    }
    if (tid == 0)
      pipe_light.wl[_light_index].reset_next_slot();
    contract_expand_dev_light (gg, LEVEL, pipe_light.wl[_light_index], pipe_light.wl[_light_index ^ 1]);
    pipe_light.wl[_light_index].swap_slots();
    gb.Sync();
    pipe.advance2();
    _light_index ^= 1;
    LEVEL++;
    if (pipe.in_wl().nitems() > WORKLISTSIZE)
    {
      break;
    }
    if (tid == 0)
      pipe_light.wl[_light_index].reset_next_slot();
    contract_expand_dev_light (gg, LEVEL, pipe_light.wl[_light_index], pipe_light.wl[_light_index ^ 1]);
    pipe_light.wl[_light_index].swap_slots();
    gb.Sync();
    pipe.advance2();
    _light_index ^= 1;
    LEVEL++;
    if (pipe.in_wl().nitems() > WORKLISTSIZE)
    {
      break;
    }
  }
  // FP: "28 -> 29;
  // FP: "25 -> 26;
  // FP: "29 -> 30;
  gb.Sync();
  // FP: "30 -> 31;
  // FP: "26 -> 27;
  // FP: "31 -> 32;
  if (tid == 0)
  {
    pipe_light.save(pipe, _light_index);
    pipe.save();
    *cl_LEVEL = LEVEL;
  }
  // FP: "35 -> 36;
  // FP: "30 -> 31;
  // FP: "36 -> 37;
}
__global__ void gg_main_pipe_1_gpu(CSRGraph gg, int LEVEL, PipeContextT<Worklist2> pipe, dim3 blocks, dim3 threads, int* cl_LEVEL)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_one;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  LEVEL = *cl_LEVEL;
  // FP: "3 -> 4;
  // FP: "2 -> 3;
  // FP: "4 -> 5;
  while (pipe.in_wl().nitems())
  {
    contract_expand <<<blocks, __tb_contract_expand>>>(gg, LEVEL, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    LEVEL++;
    if (pipe.in_wl().nitems() > WORKLISTSIZE)
    {
      break;
    }
    contract_expand <<<blocks, __tb_contract_expand>>>(gg, LEVEL, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    LEVEL++;
    if (pipe.in_wl().nitems() > WORKLISTSIZE)
    {
      break;
    }
    contract_expand <<<blocks, __tb_contract_expand>>>(gg, LEVEL, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    LEVEL++;
    if (pipe.in_wl().nitems() > WORKLISTSIZE)
    {
      break;
    }
    contract_expand <<<blocks, __tb_contract_expand>>>(gg, LEVEL, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    LEVEL++;
    if (pipe.in_wl().nitems() > WORKLISTSIZE)
    {
      break;
    }
  }
  // FP: "26 -> 27;
  // FP: "24 -> 25;
  // FP: "27 -> 28;
  if (tid == 0)
  {
    pipe.save();
    *cl_LEVEL = LEVEL;
  }
  // FP: "31 -> 32;
  // FP: "28 -> 29;
  // FP: "32 -> 33;
}
void gg_main_pipe_1_wrapper(CSRGraph& gg, int& LEVEL, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  static GlobalBarrierLifetime gg_main_pipe_1_gpu_gb_barrier;
  static bool gg_main_pipe_1_gpu_gb_barrier_inited;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  static const size_t gg_main_pipe_1_gpu_gb_residency = maximum_residency(gg_main_pipe_1_gpu_gb, __tb_gg_main_pipe_1_gpu_gb, 0);
  static const size_t gg_main_pipe_1_gpu_gb_blocks = GG_MIN(blocks.x, ggc_get_nSM() * gg_main_pipe_1_gpu_gb_residency);
  if(!gg_main_pipe_1_gpu_gb_barrier_inited) { gg_main_pipe_1_gpu_gb_barrier.Setup(gg_main_pipe_1_gpu_gb_blocks); gg_main_pipe_1_gpu_gb_barrier_inited = true;};
  // FP: "3 -> 4;
  // FP: "2 -> 3;
  // FP: "4 -> 5;
  if (false)
  {
    gg_main_pipe_1(gg,LEVEL,pipe,blocks,threads);
  }
  else
  {
    int* cl_LEVEL;
    check_cuda(cudaMalloc(&cl_LEVEL, sizeof(int) * 1));
    check_cuda(cudaMemcpy(cl_LEVEL, &LEVEL, sizeof(int) * 1, cudaMemcpyHostToDevice));
    pipe.prep();

    // gg_main_pipe_1_gpu<<<1,1>>>(gg,LEVEL,pipe,blocks,threads,cl_LEVEL);
    gg_main_pipe_1_gpu_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(gg,LEVEL,pipe,cl_LEVEL, gg_main_pipe_1_gpu_gb_barrier);
    pipe.restore();
    check_cuda(cudaMemcpy(&LEVEL, cl_LEVEL, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(cl_LEVEL));
  }
  // FP: "7 -> 8;
  // FP: "5 -> 6;
  // FP: "8 -> 9;
}
void gg_main(CSRGraph& hg, CSRGraph& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  PipeContextT<Worklist2> pipe;
  // FP: "1 -> 2;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  ApproxBitsetByte visited (hg.nnodes);
  // FP: "3 -> 4;
  // FP: "2 -> 3;
  // FP: "4 -> 5;
  kernel <<<blocks, threads>>>(gg, 0);
  // FP: "5 -> 6;
  // FP: "3 -> 4;
  // FP: "6 -> 7;
  int LEVEL = 0;
  // FP: "7 -> 8;
  // FP: "4 -> 5;
  // FP: "8 -> 9;
  pipe = PipeContextT<Worklist2>(gg.nedges > 65536 ? gg.nedges : 65536);
  pipe.in_wl().wl[0] = 0;
  pipe.in_wl().update_gpu(1);
  {
    while (pipe.in_wl().nitems())
    {
      gg_main_pipe_1_wrapper(gg,LEVEL,pipe,blocks,threads);
      while (pipe.in_wl().nitems())
      {
        pipe.out_wl().will_write();
        contract <<<blocks, __tb_contract>>>(gg, LEVEL, pipe.in_wl(), pipe.out_wl());
        pipe.in_wl().swap_slots();
        pipe.advance2();
        pipe.out_wl().will_write();
        expand <<<blocks, __tb_expand>>>(gg, pipe.in_wl(), pipe.out_wl());
        pipe.in_wl().swap_slots();
        pipe.advance2();
        pipe.out_wl().will_write();
        filter <<<blocks, __tb_filter>>>(visited, pipe.in_wl(), pipe.out_wl());
        pipe.in_wl().swap_slots();
        pipe.advance2();
        LEVEL++;
      }
    }
  }
  pipe.free();
  // FP: "23 -> 24;
  // FP: "19 -> 20;
  // FP: "24 -> 25;
}