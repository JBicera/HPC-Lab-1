#include <unistd.h>
#include <stdlib.h>
#include "listutils.h"
#include <stddef.h>
#include <math.h>
#include <omp.h>


void updateRanks(long* rankIn, long* rankOut, long* next, size_t n);
void jumpList(long* nextIn, long* nextOut, size_t n);
void swapArrays(long* arr1, long* arr2, size_t n);

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
		rank2[i] = 1;
		next1[i] = next[i];
		next2[i] = next[i];
    }
	rank1[head] = 0;
	rank2[head] = 0;
	// Iterate through log n
	for(int i = 0; i < ceil(log((double)n) / log(2.0)); i++)
	{
		updateRanks(rank1,rank2,next1, n);
		jumpList(next1,next2, n);
		swapArrays(rank1,rank2,n);
		swapArrays(next1,next2,n);
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
    #pragma omp parallel for
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
                #pragma omp atomic
                rankOut[next[i]] += rankIn[i];  // Each thread updates its part independently
            }
        }
    }
}
void jumpList(long* nextIn, long* nextOut, size_t n)
{
	#pragma omp parallel for
    for (size_t i = 0; i < n; i++) // Ensure each thread works on different indices
    {
        if(nextIn[i] != -1) // Check if nextIn[i] is not nil (-1 represents nil in your code)
        {
            nextOut[i] = nextIn[nextIn[i]];
        }
        else
            nextOut[i] = nextIn[i];
    }
}

void swapArrays(long* arr1, long* arr2, size_t n) 
{
	#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        long temp = arr1[i];
        arr1[i] = arr2[i];
        arr2[i] = temp;
    }
}

