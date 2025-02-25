#include <unistd.h>
#include "listutils.h"

__global__ void computeLocalRanksKernel(long* d_next, long* d_rank, long* d_orderedHeadNodes, long* d_sublistSizes, size_t s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s) return;

    long current = d_orderedHeadNodes[i];
    long localRank = 0;
    long sublistSize = 0;

    while (current != -1) {
        d_rank[current] = localRank;
        printf("Thread %d: Node %ld gets rank %ld\n", i, current, localRank);
        localRank++;
        sublistSize++;

        current = d_next[current];

        if (i < s - 1 && current == d_orderedHeadNodes[i + 1]) {
            break;
        }
    }

    d_sublistSizes[i] = sublistSize;
}

__global__ void computeGlobalRanksKernel(long* d_next, long* d_rank, long* d_orderedHeadNodes, long* d_sublistSizes, size_t s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s) return;

    long current = d_orderedHeadNodes[i];
    long globalHeadRank = d_rank[d_orderedHeadNodes[i]]; // The global rank of the current head node

    // Traverse the sublist and update the global rank
    while (current != -1) {
        if (current != d_orderedHeadNodes[i]) {
            d_rank[current] += globalHeadRank; // Update rank with global rank prefix sum
        }

        // Move to the next node
        current = d_next[current];

        // If we've reached the next head node, stop
        if (i < s - 1 && current == d_orderedHeadNodes[i + 1]) {
            break;
        }
    }
}


extern "C" void parallelListRanks(const long head, const long* next, long* rank, const size_t n)
{
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Get properties of GPU 0
    size_t numSMs = prop.multiProcessorCount; // Number of SMs
    size_t s = numSMs + (numSMs / 2);

    // Device pointers
    long *d_next, *d_rank, *d_orderedHeadNodes, *d_sublistSizes;

    // Assume memory is allocated and initialized elsewhere
    cudaMalloc((void**)&d_next, n * sizeof(long));
    cudaMalloc((void**)&d_rank, n * sizeof(long));

    // Copy data to device
    cudaMemcpy(d_next, next, n * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemset(d_rank, 0, n * sizeof(long)); // Initialize rank array

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

    cudaMalloc((void**)&d_orderedHeadNodes, s * sizeof(long));
    cudaMemcpy(d_orderedHeadNodes, orderedHeadNodes, s * sizeof(long), cudaMemcpyHostToDevice);

    // Allocate and copy sublistSizes
    long* sublistSizes = (long*)malloc(s * sizeof(long));
    cudaMalloc((void**)&d_sublistSizes, s * sizeof(long));
    cudaMemset(d_sublistSizes, 0, s * sizeof(long));


    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (s + threadsPerBlock - 1) / threadsPerBlock;
    
    // Step 2
    computeLocalRanksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_next, d_rank, d_orderedHeadNodes, d_sublistSizes, s);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("computeLocalRanksKernel failed: %s\n", cudaGetErrorString(err));
    }

    // Copy results back to host
    cudaMemcpy(rank, d_rank, n * sizeof(long), cudaMemcpyDeviceToHost);
    cudaMemcpy(sublistSizes, d_sublistSizes, s * sizeof(long), cudaMemcpyDeviceToHost);

    // Step 3: Sequentially update the head nodes with the accumulated rank
    long accumulatedRank = 0;  // Initialize accumulated rank to 0
    for (size_t i = 0; i < s; i++) {
        long headNode = orderedHeadNodes[i];
        // Update the rank of the head node to the accumulated rank
        rank[headNode] = accumulatedRank;
        // After updating, increment the accumulated rank by the size of the current sublist
        accumulatedRank += sublistSizes[i];
    }
    free(sublistSizes);

    // Step 4
    computeGlobalRanksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_next, d_rank, d_orderedHeadNodes, d_sublistSizes, s);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("computeGlobalRanksKernel failed: %s\n", cudaGetErrorString(err));
    }
    // Copy results back to host
    cudaMemcpy(rank, d_rank, n * sizeof(long), cudaMemcpyDeviceToHost);

    printf("Final rank array: ");
    for (size_t i = 0; i < n; i++) {
        printf("%ld ", rank[i]);
    }
    printf("\n");

    // Cleanup memory
    cudaFree(d_next);
    cudaFree(d_rank);
    cudaFree(d_orderedHeadNodes);
    cudaFree(d_sublistSizes);
}
