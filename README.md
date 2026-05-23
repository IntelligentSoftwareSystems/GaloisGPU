# LonestarGPU Benchmark Suite v3

The LonestarGPU (LSG) suite contains CUDA implementations of several
irregular algorithms that exhibit amorphous data parallelism.

The benchmarks in this repository have been moved to
[Galois](https://github.com/IntelligentSoftwareSystems/Galois/): the
instructions for running GPU benchmarks may be found in the `README`.

# INSTALLATION

You can checkout the latest release by typing (in a terminal):

```Shell
git clone -b release-4.0 https://github.com/IntelligentSoftwareSystems/GaloisGPU
```

The master branch will be regularly updated, so you may try out the latest
development code as well, by checking out the master branch:

```Shell
git clone https://github.com/IntelligentSoftwareSystems/GaloisGPU
```


## Software pre-requisites

* ModernGPU (v1.1)

[https://github.com/NVlabs/moderngpu/releases](https://github.com/NVlabs/moderngpu/releases)

ModernGPU can be downloaded as a submodule using `git submodule --init --recursive`.

Else, you may download it manually.  
Assuming LSGDIR contains the LonestarGPU source (i.e., this repository): 

```Shell
cd $LSGDIR/rt/include
ln -s path-to-mgpu-x.y/ mgpu
```

CUB is included in recent CUDA releases, and need not be installed separately.

To compile for a specific GPU architecture, make changes to the arch.mk file in the top directory.

## BUILDING


```Shell
cd $LSGDIR
make inputs # downloads the inputs required for LSG
make # compiles all applications
```

You can also recompile individual applciation ($APP) by entering $LSGDIR/apps/$APP directory.  However, this can be done only after first  *make* was performed in the $LSGDIR.

# Running

```Shell
cd $LSGDIR
make
./test /path/to/graph/file
```

#### Example:

```Shell
cd $LSGDIR
cd apps/bfs
./test /path/to/NY.gr -o output-NY
```

# Documentation

Further documentation is available at
[http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu](http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu)


# Contact Us 

For bugs, please raise an issue here at gihub using the 'Issues' tab [https://github.com/IntelligentSoftwareSystems/GaloisGPU/issues](https://github.com/IntelligentSoftwareSystems/GaloisGPU/issues).
Please send questions and comments to Galois users mailing list: [galois-users@utlists.utexas.edu](galois-users@utlists.utexas.edu). You may subscribe at
[https://utlists.utexas.edu/sympa/subscribe/galois-users](https://utlists.utexas.edu/sympa/subscribe/galois-users). 


