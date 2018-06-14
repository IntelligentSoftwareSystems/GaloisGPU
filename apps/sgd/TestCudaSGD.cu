/**
Stochastic Gradient Descent on the GPU - evaluation of different schedules on the GPU.
GPGPU8 (http://dl.acm.org/citation.cfm?id=2716289)
Copyright (C) 2015, The University of Texas at Austin. All rights reserved.

@author Anand Venkat <anandv@cs.utah.edu>
@author Rashid Kaleem <rashid.kaleem@gmail.com>

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

//#define _GOPT_DEBUG 1
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>
#include <set>
#include <string>
#include <vector>
#include <libgen.h>

#include <cuda.h>

#include "SGDAsyncEdgeCu.h"

int main(int argc, char ** args) {
   const char * fname = "../../inputs/bgg.gr";
   if (argc == 2)
      fname = args[1];
   typedef SGDAsynEdgeCudaFunctor SGDFunctorTy;
   fprintf(stderr, "===============================Starting- processing %s\n===============================", fname);
   SGDFunctorTy func(false, fname);
	func(5);      
   fprintf(stderr, "====================Terminating - processed%s================================\n", fname);
   std::cout << "Completed successfully!\n";
   return 0;
}
