#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "timer.h"
#include "listutils.h"

/* ====================================================================== */

extern void 
parallelListRanks (long head, const long* next, long* rank, size_t n);

/* ====================================================================== */

/**
 *  Given a list size 'n' and the execution time 't' in seconds,
 *  returns an estimate of the effective bandwidth (in bytes per
 *  second) of list ranking.
 */
static long double estimateBandwidth (size_t n, long double t){
  return (long double)n * (2*sizeof (long) + sizeof (long)) / t;
}


/**
 * Comparison function for qsort() in printStats.
 */
int comp (const void * elem1, const void * elem2) 
{
    long double f = *((long double*)elem1);
    long double s = *((long double*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

/**
 *  Computes the min, max, mean, and median values of an array
 *  T[0:n-1] of 'n' long doubles.
 */
static void printStats (FILE* fp, long double* T, size_t ntrials, size_t ncorrect, size_t nelements, long double ref_T){
  long double t_min, t_max, t_mean, t_median, t_speedup;

  assert (ntrials > 0);

  t_min = T[0];
  t_max = T[0];
  t_mean = T[0];
  t_median = T[0];

  for (size_t i = 1; i < ntrials; ++i) {
    if (T[i] < t_min) t_min = T[i];
    if (T[i] > t_max) t_max = T[i];
    t_mean += T[i];
  }
  t_mean /= ntrials;
  qsort(T, ntrials, sizeof(*T), comp);
  // std::nth_element(T, T + ntrials/2, T + ntrials);
  t_median = T[ntrials/2];

  //Calculate Speedup
  if (ntrials != ncorrect)
  {
    t_speedup = 0.0;
  }
  else
  {
    t_speedup = (long double)(ref_T / t_median);
  }

  fprintf(fp,"{\"ntrials\": %lu, "
              "\"ncorrect\": %lu, "
              "\"nelements\": %lu, "
              "\"min\": %Lf, "
              "\"max\": %Lf, "
              "\"mean\": %Lf, "
              "\"median\": %Lf, "
              "\"bandwidth\": %Lf, "
              "\"speedup\": %#.2Lf}\n", ntrials, ncorrect, nelements, t_min, t_max, t_mean, t_median, estimateBandwidth(nelements, t_mean), t_speedup);

}

/* ====================================================================== */
#define USAGE                                                                 \
"usage:\n"                                                                    \
"  ./metrics [options]\n"                                                        \
"options:\n"                                                                  \
"  -t [num_trials]     Number of trials (Default: 1)\n"                         \
"  -n [list_length]    Length of the list (Default: 1000000)\n"                \
"  -o [output_file]    Name of the output (Default: stdout)\n"                \
"  -h                  Show this help message\n" 

int
main (int argc, char* argv[])
{
  int option_char;
  int NTRIALS = 1;
  int N = 1000000;
  char *filename = NULL;

  while ((option_char = getopt(argc, argv, "t:n:o:h")) != -1) {
    switch (option_char) {
      case 't': 
        NTRIALS = atoi(optarg);
        break;
      case 'n':
        N = atoi(optarg);
        break;
      case 'o':
        filename = optarg;
        break;
      case 'h':
        fprintf(stderr, "%s\n", USAGE);
        exit(EXIT_SUCCESS);
        break;                      
      default:
        fprintf(stderr, "%s\n", USAGE);
        exit(EXIT_FAILURE);
    }
  }

  long double* T = (long double*) malloc(NTRIALS * sizeof(long double));

  long *next = (long*) malloc(N * sizeof(long));
  long *par_rank = (long*) malloc(N * sizeof(long));
  long *seq_rank = (long*) malloc(N * sizeof(long));
  size_t ncorrect = 0;

  long *ref_next = (long*) malloc(N * sizeof(long));
  long *ref_rank = (long*) malloc(N * sizeof(long));
  long double ref_T = 0.0;
  long ref_head;
  stopwatch_t ref_watch;

  if (NTRIALS > 0)
  {
    //Get the serial reference time for speedup
    initRandomList(ref_next, N);
    ref_head = seqFindHead(ref_next, N);

    ref_watch = stopwatch_start();
    seqListRanks(ref_head, ref_next, ref_rank, N);
    ref_T = stopwatch_stop(ref_watch);
  }

  free(ref_next);
  free(ref_rank);

  for(int i = 0; i < NTRIALS; i++){

    initRandomList(next, N);

    long head = seqFindHead(next, N);

    seqListRanks(head, next, seq_rank, N);

    stopwatch_t watch = stopwatch_start();
    parallelListRanks(head, next, par_rank, N);
    T[i] = stopwatch_stop(watch);

    if( memcmp(seq_rank, par_rank, N * sizeof(long)) == 0)
      ncorrect++;
  }

  if (!filename)
    printStats(stdout, T, NTRIALS, ncorrect, N, ref_T);
  else{
    FILE *fp;
    fp = fopen(filename, "w");
    printStats(fp, T, NTRIALS, ncorrect, N, ref_T);
    fclose(fp);
  }

  free(next);
  free(par_rank);
  free(seq_rank);
  free(T);

  return EXIT_SUCCESS;
}
