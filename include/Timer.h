#pragma once
/* Author: Sreepathi Pai <sreepai@ices.utexas.edu> */

#include <time.h>
#include <unistd.h>
#include <errno.h>
#if !(_POSIX_TIMERS > 0)
#error "POSIX timers not available"
#endif

#ifdef _POSIX_MONOTONIC_CLOCK
/* should probably use CLOCK_MONOTONIC_RAW */
#define CLOCKTYPE CLOCK_MONOTONIC 
#define SCLOCKTYPE "CLOCK_MONOTONIC"
#else
#warning "CLOCK_MONOTONIC is unavailable, using CLOCK_REALTIME"
#define CLOCKTYPE CLOCK_REALTIME
#define SCLOCKTYPE "CLOCK_REALTIME"
#endif


#define NANOSEC 1000000000LL

class Timer {
  char const *name;
  struct timespec begin, end;
  bool active, valid;
  unsigned long long last;
  unsigned long long total;

 public:
  
  Timer(const char *timer_name) {
    name = timer_name;
    active = false;
    valid = false;
    total = 0;
  }

  unsigned long long normalize(const struct timespec &t) const {
    return t.tv_sec * NANOSEC + t.tv_nsec;
  }

  void reset() {
    assert(!active);
    total = 0;
    last = 0;
  }

  void start() {
    assert(!active);
    active = true;
    valid = false;
    if(clock_gettime(CLOCKTYPE, &begin) == -1) {
      if(errno == EINVAL) {
	fprintf(stderr, SCLOCKTYPE " not available.\n");
	// exit?
      }
    }
  }

  void stop() {
    assert(active);

    if(clock_gettime(CLOCKTYPE, &end) == -1) {
      if(errno == EINVAL) {
	fprintf(stderr, SCLOCKTYPE " not available.\n");
	// exit?
      }
    }

    // assert(normalize(end) > normalize(begin) // paranoid level 2

    last = normalize(end) - normalize(begin);
    total += last;
    active = false;
    valid = true;
  }

  unsigned long long duration() const {
    return last;
  }

  unsigned long long total_duration() const {
    return total;
  }
};

