#include <unistd.h>
#include "listutils.h"

// Debugging kernel: set all elements of an array to 1
__global__ void setAllToOneKernel(long* arr, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = 1;
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
    long *dNext, *dRank, *dOrderedHeadNodes, *dSublistSizes;

    // Assume memory is allocated and initialized elsewhere
    cudaMalloc((void**)&dNext, n * sizeof(long));
    cudaMalloc((void**)&dRank, n * sizeof(long));

    // Copy data to device
    cudaMemcpy(dNext, next, n * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemset(dRank, 0, n * sizeof(long)); // Initialize rank array

    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        setAllToOneKernel<<<blocksPerGrid, threadsPerBlock>>>(dRank, n);
        cudaDeviceSynchronize();
        // Optionally, copy back and print to check:
        long* debugRank = (long*)malloc(n * sizeof(long));
        cudaMemcpy(debugRank, dRank, n * sizeof(long), cudaMemcpyDeviceToHost);
        printf("Debug: dRank after setAllToOneKernel: ");
        for (size_t i = 0; i < n; i++) {
            printf("%ld ", debugRank[i]);
        }
        printf("\n");
        free(debugRank);
    }

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
    
    // Cleanup memory
    cudaFree(dNext);
    cudaFree(dRank);
    cudaFree(dOrderedHeadNodes);
    cudaFree(dSublistSizes);
    
}
