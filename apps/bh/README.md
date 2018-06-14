## DESCRIPTION

This benchmark simulates the gravitational forces acting on a galactic cluster using the Barnes-Hut n-body algorithm. 
The positions and velocities of the n galaxies are initialized according to the empirical Plummer model. 
The program calculates the motion of each galaxy through space for a number of time steps. 
The data parallelism in this algorithm arises primarily from the independent force calculations.



## COMPILE

Run make in the root directory or in the source folder (e.g. apps/bh)

## RUN

Execute as: bh <bodies> <timesteps> <deviceid>
e.g., ./bh 50000 2 0
