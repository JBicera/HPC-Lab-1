#include <unistd.h>
#include "listutils.h"

void parallelListRanks (long head, const long* next, long* rank, size_t n)
{
	/* Basic serial implementation */

	long r = 0;

	while(head != NIL)
	{
		rank[head] = r++;
		head = next[head];
	}
}
