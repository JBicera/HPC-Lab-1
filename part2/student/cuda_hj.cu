#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printArray(const char* name, long* arr, size_t n) {
    printf("%s: ", name);
    for (size_t i = 0; i < n; i++) {
        printf("%ld ", arr[i]);
    }
    printf("\n");
}

// CUDA kernel to compute local ranks for each node in the sublists
__global__ void computeLocalRanks(long *rank, const long *next, const long *orderedHeadNodes, long *sublistSizes, size_t s, size_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Thread index

    if (tid < s) {
        long current = orderedHeadNodes[tid];
        long localRank = 0;
        long sublistSize = 0;

        // Traverse the sublist starting from the current head node
        while (current != -1) {
            rank[current] = localRank;  // Assign local rank
            localRank++;
            sublistSize++;

            current = next[current];

            // Stop if next head node is reached
            if (current == orderedHeadNodes[tid + 1]) {
                break;
            }
        }

        sublistSizes[tid] = sublistSize;  // Store sublist size
    }
}

// CUDA kernel to update global ranks based on head nodes' ranks
__global__ void updateGlobalRanks(long *rank, const long *next, const long *orderedHeadNodes, size_t s)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < s) {
        long current = orderedHeadNodes[tid];
        long globalHeadRank = rank[orderedHeadNodes[tid]];  // Rank of the current head node

        // Traverse the sublist again and update global rank
        while (current != -1) {
            if (current != orderedHeadNodes[tid]) {
                rank[current] += globalHeadRank;  // Add global head rank to the current node's rank
            }

            current = next[current];

            // Stop when we reach the next head node
            if (current == orderedHeadNodes[tid + 1]) {
                break;
            }
        }
    }
}

// Function to convert your OpenMP-based parallelListRanks into CUDA
void parallelListRanks(long head, const long* next, long* rank, size_t n)
{
    // Calculate number of sublists, s
    size_t numP = 256;  // Default number of threads per block
    size_t s = numP + (numP / 3); // Choose s > P for load balancing
    long *headNodes = (long*)malloc(s * sizeof(long));
    headNodes[0] = head;  // Ensure true head is included

    // Step 1: Randomly choose head nodes for s sublists.
    srand(42);
    int* used = (int*)calloc(n, sizeof(int));  // Track used nodes
    used[head] = 1;  // True head is used
    for (size_t i = 1; i < s; i++) {
        size_t idx;
        do {
            idx = rand() % n;
        } while (used[idx] == 1);
        headNodes[i] = idx;
        used[idx] = 1;
    }
    free(used);

    // Step 2: Create orderedHeadNodes list
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
    free(headNodes);

    // Step 3: Memory allocation on device
    long *d_rank, *d_next, *d_orderedHeadNodes, *d_sublistSizes;
    cudaMalloc((void**)&d_rank, n * sizeof(long));
    cudaMalloc((void**)&d_next, n * sizeof(long));
    cudaMalloc((void**)&d_orderedHeadNodes, s * sizeof(long));
    cudaMalloc((void**)&d_sublistSizes, s * sizeof(long));

    cudaMemcpy(d_rank, rank, n * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_next, next, n * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_orderedHeadNodes, orderedHeadNodes, s * sizeof(long), cudaMemcpyHostToDevice);

    // Step 4: Parallel traversal to compute local ranks (kernel launch)
    int blockSize = 256;
    int numBlocks = (s + blockSize - 1) / blockSize;
    computeLocalRanks<<<numBlocks, blockSize>>>(d_rank, d_next, d_orderedHeadNodes, d_sublistSizes, s, n);

    cudaDeviceSynchronize();

    // Step 5: Sequentially update the head nodes with the accumulated rank (can be done on host)
    long* sublistSizes = (long*)malloc(s * sizeof(long));
    cudaMemcpy(sublistSizes, d_sublistSizes, s * sizeof(long), cudaMemcpyDeviceToHost);

    long accumulatedRank = 0;
    for (size_t i = 0; i < s; i++) {
        long headNode = orderedHeadNodes[i];
        rank[headNode] = accumulatedRank;
        accumulatedRank += sublistSizes[i];
    }
    free(sublistSizes);

    // Step 6: Update global ranks for nodes in each sublist (kernel launch)
    updateGlobalRanks<<<numBlocks, blockSize>>>(d_rank, d_next, d_orderedHeadNodes, s);
    cudaDeviceSynchronize();

    // Step 7: Free device memory
    cudaFree(d_rank);
    cudaFree(d_next);
    cudaFree(d_orderedHeadNodes);
    cudaFree(d_sublistSizes);

    free(orderedHeadNodes);
}
