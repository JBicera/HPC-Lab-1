#include <unistd.h>
#include "listutils.h"

// CUDA kernel to compute local ranks and sublist sizes
__global__ void parallelListRanksKernel(const long* next, long* rank, const long* orderedHeadNodes, long* sublistSizes, size_t s, size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < s) {
        long current = orderedHeadNodes[idx];
        long localRank = 0;
        long sublistSize = 0;

        while (current != -1) {
            rank[current] = localRank;
            localRank++;
            sublistSize++;

            current = next[current];

            // If we have reached the next head node, break
            if (idx < s - 1 && current == orderedHeadNodes[idx + 1]) {
                break;
            }
        }

        sublistSizes[idx] = sublistSize;
    }
}

// CUDA kernel to update global ranks based on head node ranks
__global__ void parallelGlobalRankUpdateKernel(const long* next, long* rank, const long* orderedHeadNodes, size_t s, size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < s) {
        long current = orderedHeadNodes[idx];
        long globalHeadRank = rank[orderedHeadNodes[idx]];

        while (current != -1) {
            if (current != orderedHeadNodes[idx]) {
                rank[current] += globalHeadRank;
            }

            current = next[current];

            // If we've reached the next head node, break
            if (idx < s - 1 && current == orderedHeadNodes[idx + 1]) {
                break;
            }
        }
    }
}

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

    // Copy ordered head nodes to the device
    cudaMemcpy(dOrderedHeadNodes, orderedHeadNodes, s * sizeof(long), cudaMemcpyHostToDevice);

    free(inHead);
    free(headNodes);

    // Step 2: Parallel traversal to compute local ranks for each node in the sublists
    cudaMemset(dSublistSizes, 0, s * sizeof(long)); // Initialize sublist sizes

    // Launch kernel to compute local ranks for sublists
    int blockSize = 256; // Define block size
    int numBlocks = (s + blockSize - 1) / blockSize; // Calculate number of blocks
    parallelListRanksKernel<<<numBlocks, blockSize>>>(dNext, dRank, dOrderedHeadNodes, dSublistSizes, s, n);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    // Check for errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Step 3: Sequentially update the head nodes with the accumulated rank
    long* sublistSizes = (long*)malloc(s * sizeof(long));
    cudaMemcpy(sublistSizes, dSublistSizes, s * sizeof(long), cudaMemcpyDeviceToHost);

    long accumulatedRank = 0;
    for (size_t i = 0; i < s; i++) {
        long headNode = orderedHeadNodes[i];
        rank[headNode] = accumulatedRank;
        accumulatedRank += sublistSizes[i];
    }
    free(sublistSizes);

    // Step 4: Update global ranks for nodes in each sublist
    parallelGlobalRankUpdateKernel<<<numBlocks, blockSize>>>(dNext, dRank, dOrderedHeadNodes, s, n);
    cudaDeviceSynchronize();

    // Check for errors during kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Cleanup memory
    cudaFree(dNext);
    cudaFree(dRank);
    cudaFree(dOrderedHeadNodes);
    cudaFree(dSublistSizes);
}


