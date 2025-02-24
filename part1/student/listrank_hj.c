#include <unistd.h>
#include "listutils.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>


void printArray(const char* name, long* arr, size_t n) {
    printf("%s: ", name);
    for (size_t i = 0; i < n; i++) {
        printf("%ld ", arr[i]);
    }
    printf("\n");
}

void printConstArray(const char* name, const long* arr, size_t n) {
    printf("%s: ", name);
    for (size_t i = 0; i < n; i++) {
        printf("%ld ", arr[i]);
    }
    printf("\n");
}

// head = index of true head
// next = pointer to next array (linked list pointers; -1 means end)
// rank = array to store final ranks (of size n)
// n = number of nodes in the list
void parallelListRanks (long head, const long* next, long* rank, size_t n)
{

    // Calculate number of sublists, s, based on number of processors.
    size_t numP = omp_get_num_procs();
    size_t s = numP + (numP / 3); // Choose s > P for load balancing

    long* headNodes = (long*)malloc(s * sizeof(long));
    headNodes[0] = head; // Ensure the true head is included

    // Step 1: Randomly choose head nodes for s sublists.
    srand(42); // reproducible randomness
    int* used = (int*)calloc(n, sizeof(int)); // track which indices are chosen
    used[head] = 1; // true head is used
    for (size_t i = 1; i < s; i++) {
        size_t idx;
        do {
            idx = rand() % n;
        } while (used[idx] == 1);
        headNodes[i] = idx;
        used[idx] = 1;
    }
    free(used);

    // Create temporary bit map to show which nodes are head nodes
    char* inHead = (char*)calloc(n, sizeof(char));
    for (size_t i = 0; i < s; i++) {
        inHead[ headNodes[i] ] = 1;
    }
    // Create an orderedHeadNodes list 
    long* orderedHeadNodes = (long*)malloc(s * sizeof(long));
    size_t count = 0;
    long current = head;
    while (current != -1 && count < s) {
        if (inHead[current]) {  // O(1) membership check
            orderedHeadNodes[count] = current;
            count++;
        }
        current = next[current];
    }
    free(headNodes);

    
    // Step 2: Parallel traversal to compute local ranks for each node in the sublists
    long *sublistSizes = (long *)malloc(s * sizeof(long));
    memset(sublistSizes, 0, s * sizeof(long));  // Initialize all sublist sizes to 0
    #pragma omp parallel for
    for (size_t i = 0; i < s; i++) {
        long current = orderedHeadNodes[i];
        long localRank = 0;
        long sublistSize = 0;

        // Traverse the sublist starting from the current head node
        while (current != -1) {
            // Assign the local rank to the current node
            rank[current] = localRank;
            localRank++;
            sublistSize++;

            // Move to the next node in the list
            current = next[current];

            // If we have reached the next head node (and it's not the last sublist)
            if (current == orderedHeadNodes[i + 1]) {
                break;
            }
        }

        // Store the sublist size for later use in Step 2
        sublistSizes[i] = sublistSize;  // Keep track of sublist sizes
    }

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

    
    //Step 4: Update global ranks for nodes in each sublist
    #pragma omp parallel for
    for (size_t i = 0; i < s; i++) {
        long current = orderedHeadNodes[i];
        long globalHeadRank = rank[orderedHeadNodes[i]]; // The global rank of the current head node

        // Traverse the sublist again and update the global rank based on the head node's global rank
        while (current != -1) 
        {
            if (current != orderedHeadNodes[i]) 
            {
                rank[current] += globalHeadRank; // Update rank with global rank prefix sum
            }

            // Move to the next node
            current = next[current];

            // If we've reached the next head node, stop
            if (current == orderedHeadNodes[i + 1]) {
                break;
            }
        }
    }

    free(orderedHeadNodes);
}
