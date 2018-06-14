# LonestarGPU Benchmark Suite v3

The LonestarGPU (LSG) suite contains CUDA implementations of several
irregular algorithms that exhibit amorphous data parallelism.

## INSTALLATION


### Software pre-requisites

* CUB (v1.3.1 or later)

https://github.com/NVlabs/cub


* ModernGPU (v1.1 or later)

https://github.com/NVlabs/moderngpu/releases

You will need to download CUB and Mgpu in the root directory:

$ cd $LSGDIR
$ ln -s path-to-cub-x.y/ cub
$ ln -s path-to-mgpu-x.y/ mgpu

### BUILDING

Assuming you're in $LSGDIR:

$ make inputs # downloads the inputs required for LSG

$ make # compiles all applications

## Running

$ make
$ ./test /path/to/graph/file

Example:

$ cd $apps/bfs
$ ./test /path/to/NY.gr -o output-NY


