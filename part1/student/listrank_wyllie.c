#include <unistd.h>
#include <stdlib.h>
#include "listutils.h"
#include <stddef.h>
#include <math.h>
#include <omp.h>
#include <string.h>



void updateRanks(long* rankIn, long* rankOut, long* next, size_t n);
void jumpList(long* nextIn, long* nextOut, size_t n);

// Helper to just print results
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

	// Initialize all values
    for (long i = 0; i < n; i++) {
        rank1[i] = 1;
		next1[i] = next[i];
    }
	rank1[head] = 0;

    // Copy instead of looping
    memcpy(rank2, rank1, n * sizeof(long));  
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

    // Copy result to original rank
	memcpy(rank, rank1, n * sizeof(long));

	// Free memory
	free(rank1);
	free(rank2);
	free(next1);
	free(next2);
}


void updateRanks(long* rankIn, long* rankOut, long* next, size_t n)
{
    // Initialize rankOut in parallel
    #pragma omp parallel for 
    for (size_t i = 0; i < n; i++)
        rankOut[i] = rankIn[i];
    // Update rank in parallel
    #pragma omp parallel for 
    for (size_t i = 0; i < n; i++) 
        if (next[i] != -1) 
            rankOut[next[i]] += rankIn[i];  
}

void jumpList(long* nextIn, long* nextOut, size_t n)
{
    // Jump ranks in parallel
	#pragma omp parallel for 
    for (size_t i = 0; i < n; i++) {
        if (nextIn[i] != -1) 
            nextOut[i] = nextIn[nextIn[i]];
        else 
            nextOut[i] = nextIn[i];
        
    }

}
