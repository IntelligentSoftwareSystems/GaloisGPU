## DESCRIPTION

SGD is an iterative algorithm that computes feature vectors by making a number of sweeps over the bipartite graph. The vectors are initialized to some arbitrary values. In each sweep, all edges (u, m) are visited. If the inner-product of the vectors on nodes u and m is not equal to the weight on edge (u,m), the difference is used to update the two feature vectors. Sweeps are terminated when some heuristic measure of convergence is reached. In a given graph, a set of edges is said to constitute a matching if no two edges in that set have a node in common. If the graph is viewed as an adjacency matrix, entries along the diagonals of the matrix can be processed concurrently as they do not share any end-points. This program uses a class of preprocessing techniques called diagonal matchings, which have lower preprocessing time. The BlkDiag schedule reduces the size of the matrix by blocking along both dimensions. This reduced matrix has a reduced number of diagonals.
The detailed algorithm description is [http://www.cs.utexas.edu/~rashid/public/ipdps2016.pdf](in this paper).

## COMPILE

Simply run make in the root directory or in the source code directory (e.g. apps/sgd)

## Run

./blk_diag path-to-input


