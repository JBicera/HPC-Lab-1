#include <unistd.h>
#include "listutils.h"

__global__ void computeLocalRanksKernel(long* dNext, long* dRank, long* dOrderedHeadNodes, long* dSublistSizes, size_t s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s) return;

    long current = dOrderedHeadNodes[i];
    long localRank = 0;
    long sublistSize = 0;

    while (current != -1) {
        dRank[current] = localRank;
        printf("Thread %d: Node %ld gets rank %ld\n", i, current, localRank);
        localRank++;
        sublistSize++;

        current = dNext[current];

        if (i < s - 1 && current == dOrderedHeadNodes[i + 1]) {
            break;
        }
    }

    dSublistSizes[i] = sublistSize;
}

extern "C" void parallelListRanks(const long head, const long* next, long* rank, const size_t n)
{
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Get properties of GPU 0
    size_t numSMs = prop.multiProcessorCount; // Number of SMs
    size_t s = numSMs + (numSMs / 2);

    // Device pointers
    long *dNext, *dRank, *dOrderedHeadNodes, *dSublistSizes;

    // Assume memory is allocated and initialized elsewhere
    cudaMalloc((void**)&dNext, n * sizeof(long));
    cudaMalloc((void**)&dRank, n * sizeof(long));

    // Copy data to device
    cudaMemcpy(dNext, next, n * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemset(dRank, 0, n * sizeof(long)); // Initialize rank array

    // Step 1: Select head nodes
    long* headNodes = (long*)malloc(s * sizeof(long));
    headNodes[0] = head;  // Ensure true head is included
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
    // Create orderedHeadNodes
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
    for (size_t i = 0; i < s; i++) {
        printf("%ld ", orderedHeadNodes[i]);
    }
    printf("\n");

    free(inHead);
    free(headNodes);
    /*
    cudaMalloc((void**)&dOrderedHeadNodes, s * sizeof(long));
    cudaMemcpy(dOrderedHeadNodes, orderedHeadNodes, s * sizeof(long), cudaMemcpyHostToDevice);

    // Allocate and copy sublistSizes
    long* sublistSizes = (long*)malloc(s * sizeof(long));
    cudaMalloc((void**)&dSublistSizes, s * sizeof(long));
    cudaMemset(dSublistSizes, 0, s * sizeof(long));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (s + threadsPerBlock - 1) / threadsPerBlock;
    
    // Step 2
    computeLocalRanksKernel<<<blocksPerGrid, threadsPerBlock>>>(dNext, dRank, dOrderedHeadNodes, dSublistSizes, s);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("computeLocalRanksKernel failed: %s\n", cudaGetErrorString(err));
    }
    

    // Copy results back to host
    cudaMemcpy(rank, dRank, n * sizeof(long), cudaMemcpyDeviceToHost);
    cudaMemcpy(sublistSizes, dSublistSizes, s * sizeof(long), cudaMemcpyDeviceToHost);
    
    printf("Final rank array: ");
    for (size_t i = 0; i < n; i++) {
        printf("%ld ", rank[i]);
    }
    printf("\n");

    // Cleanup memory
    free(sublistSizes);
    cudaFree(dNext);
    cudaFree(dRank);
    cudaFree(dOrderedHeadNodes);
    cudaFree(dSublistSizes);
    */
}
