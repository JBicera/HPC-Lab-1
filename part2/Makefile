CC = gcc
NVCC = nvcc
CFLAGS = -Iutils -O0 -g -fopenmp -std=gnu99 -Wall -Werror
NVCFLAGS = -Iutils -O0 -g -std=c++11
LDFLAGS = -lm

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -o $@ -c $<

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

ifeq ($(strip $(IMPL)), )
 IMPL=hj
endif

all: correctness metrics

correctness: utils/correctness.o student/cuda_$(IMPL).o utils/listutils.o
	$(NVCC) $(NVCFLAGS) -o $@ $^ $(LDFLAGS)

metrics: utils/metrics.o student/cuda_$(IMPL).o utils/listutils.o utils/timer.o
	$(NVCC) $(NVCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *~ core correctness metrics *.o student/*.o utils/*.o
