#pragma once

/* hand-written */

#ifdef __KEPLER__

// thread count
#define THREADS1 1024  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 768
#define THREADS4 128
#define THREADS5 1024
#define THREADS6 1024

// block count = factor * #SMs
#define FACTOR1 2
#define FACTOR2 2
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 4  /* must all be resident at the same time */
#define FACTOR5 2
#define FACTOR6 2

static const char *TUNING_PARAMETERS = "THREADS1 1024\nTHREADS2 1024\nTHREADS3 768\nTHREADS4 128\nTHREADS5 1024\nTHREADS6 1024\nFACTOR1 2\nFACTOR2 2\nFACTOR3 1\nFACTOR4 4\nFACTOR5 2\nFACTOR6 2\n";

#else

// thread count
#define THREADS1 512  /* must be a power of 2 */
#define THREADS2 512
#define THREADS3 128
#define THREADS4 64
#define THREADS5 256
#define THREADS6 1024

// block count = factor * #SMs
#define FACTOR1 3
#define FACTOR2 3
#define FACTOR3 6  /* must all be resident at the same time */
#define FACTOR4 6  /* must all be resident at the same time */
#define FACTOR5 5
#define FACTOR6 1

static const char *TUNING_PARAMETERS = "THREADS1 512\nTHREADS2 512\nTHREADS3 128\nTHREADS4 64\nTHREADS5 256\nTHREADS6 1024\nFACTOR1 3\nFACTOR2 3\nFACTOR3 6\nFACTOR4 6\nFACTOR5 5\nFACTOR6 1\n";

#endif
