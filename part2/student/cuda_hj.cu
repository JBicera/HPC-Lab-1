#include <unistd.h>
#include "listutils.h"

extern "C" void parallelListRanks(const long head, const long* next, long* rank, const size_t n)
{
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Get properties of GPU 0
    size_t numSMs = prop.multiProcessorCount; // Number of SMs
    size_t s = numSMs + (numSMs / 2); // Number of sublists

    // Device pointers
    long *dNext, *dRank, *dOrderedHeadNodes, *dSublistSizes;

    // Allocate memory on the device
    cudaMalloc((void**)&dNext, n * sizeof(long));
    cudaMalloc((void**)&dRank, n * sizeof(long));
    cudaMalloc((void**)&dOrderedHeadNodes, s * sizeof(long));
    cudaMalloc((void**)&dSublistSizes, s * sizeof(long));

    // Copy data to the device
    cudaMemcpy(dNext, next, n * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemset(dRank, 0, n * sizeof(long)); // Initialize rank array on the device

    // Step 1: Select head nodes
    long* headNodes = (long*)malloc(s * sizeof(long));
    headNodes[0] = head; // Ensure true head is included
    srand(42);
    int* used = (int*)calloc(n, sizeof(int));
    used[head] = 1;
    for (size_t i = 1; i < s; i++) {
        size_t idx;
        do {
            idx = rand() % n;
        } while (used[idx] == 1);
        headNodes[i] = idx;
        used[idx] = 1;
    }
    free(used);

    // Create orderedHeadNodes on host
    char* inHead = (char*)calloc(n, sizeof(char));
    for (size_t i = 0; i < s; i++) {
        inHead[headNodes[i]] = 1;
    }
    long* orderedHeadNodes = (long*)malloc(s * sizeof(long));
    size_t count = 0;
    long current = head;
    while (current != -1 && count < s) {
        if (inHead[current]) {
            orderedHeadNodes[count] = current;
            count++;
        }
        current = next[current];
    }

    printf("Ordered head nodes: ");
    fflush(stdout); // Ensure output is flushed immediately
    for (size_t i = 0; i < s; i++) {
        printf("%ld ", orderedHeadNodes[i]);
        fflush(stdout); // Flush after each print to avoid buffering issues
    }
    printf("\n");
    fflush(stdout);

    free(inHead);
    free(headNodes);

    // Cleanup memory
    cudaFree(dNext);
    cudaFree(dRank);
    cudaFree(dOrderedHeadNodes);
    cudaFree(dSublistSizes);
}
