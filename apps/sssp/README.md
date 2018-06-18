## DESCRIPTION

SSSP finds the shortest paths from a single source to all other vertices. This program is based on the SSSP near-far algorithm described in (https://people.csail.mit.edu/jshun/6886-s18/papers/DBGO14.pdf)[this paper].
In SSSP-nf ather than processing all vertices in the work queue, we can instead select a subset of  vertices  to  process  based  on  some  scoring  heuristic.  A simple and effective heuristic is to select a splitting distance
and then process only those vertices less than that distance. 
For more information please see the paper. 


## COMPILE

Run make in the root directory or the source directory (e.g. apps/sssp)

## RUN

./test path-to-graph -o output -d delta










