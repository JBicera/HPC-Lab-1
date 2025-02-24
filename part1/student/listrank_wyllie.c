#include <unistd.h>
#include <stdlib.h>
#include "listutils.h"
#include <stddef.h>
#include <math.h>
#include <omp.h>
#include <string.h>



void updateRanks(long* rankIn, long* rankOut, long* next, size_t n);
void jumpList(long* nextIn, long* nextOut, size_t n);

void printArray(const char* name, long* arr, size_t n) {
    printf("%s: ", name);
    for (size_t i = 0; i < n; i++) {
        printf("%ld ", arr[i]);
    }
    printf("\n");
}

// head = index of head
// next = pointer to next array
// rank = array where ranks of elements will be stored
// n = number of elements in list
void parallelListRanks (long head, const long* next, long* rank, size_t n)
{
	long* rank1 = (long*)malloc(n*sizeof(long));
	long* rank2 = (long*)malloc(n*sizeof(long));
	long* next1 = (long*)malloc(n*sizeof(long));
	long* next2 = (long*)malloc(n*sizeof(long));
	// Set all values
    for (long i = 0; i < n; i++) {
        rank1[i] = 1;
		next1[i] = next[i];
    }
	rank1[head] = 0;
    memcpy(rank2, rank1, n * sizeof(long));  // Copy instead of looping
    memcpy(next2, next, n * sizeof(long));
	// Iterate through log n
    int numIterations = ceil(log2((double)n));
    long* temp;
	for(int i = 0; i < numIterations; i++)
	{
		updateRanks(rank1,rank2,next1, n);
		jumpList(next1,next2, n);

        // Swap arrays
        temp = rank1; rank1 = rank2; rank2 = temp;
        temp = next1; next1 = next2; next2 = temp;

	}
    
	memcpy(rank, rank1, n * sizeof(long));
	// Free memory
	free(rank1);
	free(rank2);
	free(next1);
	free(next2);
}
void updateRanks(long* rankIn, long* rankOut, long* next, size_t n)
{
    // Step 1: Parallelize the first loop
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        rankOut[i] = rankIn[i];
    }

    // Step 2: Parallelize the second loop, dividing the work by thread number
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();  // Get thread number
        int num_threads = omp_get_num_threads();  // Get total number of threads

        // Each thread works on a portion of the array
        for (size_t i = thread_id; i < n; i += num_threads) {
            if (next[i] != -1) {  // If there is a valid next index
                rankOut[next[i]] += rankIn[i];  // Each thread updates its part independently
            }
        }
    }
}
void jumpList(long* nextIn, long* nextOut, size_t n)
{
	#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        if (nextIn[i] != -1) {
            nextOut[i] = nextIn[nextIn[i]];
        } else {
            nextOut[i] = nextIn[i];
        }
    }

}
