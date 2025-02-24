# Lab 1: List Ranking -- OMP and CUDA

In this lab, you will implement Wyllie's list ranking algorithm as discussed in the videos, plus another algorithm designed by Helman and JaJa.  The efficiency of your code will be compared to a baseline serial implementation to help determine your grade.

## Getting Started
Begin by obtaining the starter code from the github repository:

<pre><code>
	git clone https://github.gatech.edu/omscse6220/lab1sp25.git
</code></pre>

Optionally, you may choose to use a git hosting service for your code.  As always, please **do not make your repository public** on Github or any other hosting service.  Anyone will be able to see it.  If you feel the need to use a hosting service for your project, **please keep your repository private**.  Your GT account allows you to create free private repositories on [the GT github server](https://github.gatech.edu).

## How The Project Is Setup
The project is structured into two parts:

* OpenMP (part1 subdirectory)
* CUDA (part2 subdirectory)

Each subdirectory is structured much like lab0:

```sh
   [part]
   ---> Makefile
   ---> grader.sh
   ---> timing_*.sub
   ---> utils
   ---> student
   ...
```

## Programming
### OpenMP (Part 1)
You should only modify the contents of *listrank_wyllie.c* and *listrank_hj.c* in the *student* directory. Do not modify any other files, and do not modify the function signature. You may add helper functions as long as they are completely self-contained within the respective source file. You may also include additional headers as long as they do not require a change to the *Makefile*.

First implement the parallel list ranking algorithms using OpenMP, an API for shared memory parallel applications.There is a wealth of documentation about OpenMP online.  Here are a few resources to help you get started

*  [LLNL](https://hpc-tutorials.llnl.gov/openmp/)
*  [Wikipedia](https://en.wikipedia.org/wiki/OpenMP)
*  [OpenMP.org](https://www.openmp.org/resources/)

You should allow OpenMP to decide how many threads are available rather than choosing for yourself.  Thus, a simple

<pre><code>
	#pragma omp parallel
</code></pre>

directive will be sufficient for the purposes of this lab.

Your first task is to implement Wyllie's algorithm, which is described in the lesson videos, in the file *student/listrank_wyllie.c*.

Next, you are to implement the Helman-JaJa algorithm for list ranking in the *student/listrank_hj.c* file.  We provide the [the original paper](https://github.gatech.edu/omscse6220/class_readings/raw/master/required/hj_listrank.pdf), though you may find that the summary below is sufficient:

1. Sequentially, partition the input list into *s* sublists, by randomly choosing *s* sublist head nodes.
2. In parallel, traverse each sublist computing the list ranking of each node within the sublists.
3. Sequentially, compute the list ranking of the head nodes overall (by doing a prefix sum of the head node ranks).
4. In parallel, traverse the sublists again to convert the sublist ranking to the complete list ranking (by adding the previously determined prefix sum values).

### CUDA (Part 2)
Next, you are to implement Helman-JaJa using CUDA in *student/cuda_hj.cu*. Do not modify any other files, and do not modify the function signature. You may add helper functions as long as they are completely self-contained within the respective source file. You may also include additional headers as long as they do not require a change to the *Makefile*.

You will need to do this development on the cluster **unless** you have CUDA installed and configured locally.

To connect to a GPU node on the cluster and setup your environment you would do:

<pre><code>
	./pace_interactive.sh
	[wait till the node is acquired]
	cd $SLURM_SUBMIT_DIR
	source pace_env.sh
</code></pre>

Please make sure to save your work frequently in the event your session times out.

To help you get started please make sure you've completed the **Scaling with GPUs** module on Canvas.

## Compilation
In addition to the starter code discussed above, the repository also contains a number of other files to help you test and measure the performance of your code.  The *correctness* utility contains a simple test to see if your code is correct.  By default, the *Makefile* will compile the Wyllie variant. To have it compile the Helman-JaJa version, use the command:

<pre><code>
	make correctness IMPL=hj
</code></pre>

Note that the value for the variable IMPL given on the command line overrides the one given in the *Makefile*.

## Testing
### Correctness
We provide an automatic grading script (**grader.sh**) that will test your solution against various pre-defined problem sizes. You can run this script as many times as you would like. This script calls the *correctness* utility mentioned above with some predefined inputs.

You can run it like so:

<pre><code>
	./grader.sh
</code></pre>

The provided tests will constitute a significant portion of your grade. We will re-run them after the deadline on our end.

You may also find the following tools useful:

* GDB
* Valgrind

**NOTE: The part2 grader script will only return passing results on the cluster GPU nodes (or other compatible systems).**

### Performance
You should measure the performance of your code on the PACE cluster. The *metrics* utility performs some of the same tests on which your code will be evaluated.  Use the IMPL command line parameter to change which implementation of the parallelListRanks is compiled.  Thus, if you wanted to time your Helman-JaJa OpenMP algorithm, you might implement it in a file listrank_hj.c and compile it with:

<pre><code>
	make metrics IMPL=hj
</code></pre>

The file *timing_hj.sub* serves as an example for how to use sbatch for the Helman-JaJa algorithm.  Thus, to measure the performance of your code you should run

<pre><code>
	sbatch timing_hj.sub
</code></pre>

from the PACE login node.

## Submission

Lab submissions for this class will be through Canvas under the appropriate assignment.

The files you need to submit for this lab are:

```sh
   part1/student/listrank_wyllie.c
   part1/student/listrank_hj.c
   part2/student/cuda_hj.cu
```
These files should exist in the **student** directories of your lab folder as the paths suggest.

**Please include each .c/.cu file in a single level zip file named lab1.zip that contains no folders.**

Do **not** submit or modify any other files. Any deviation may result in a significant point deduction.

## Grading

Your grade on this lab will consist of the following:

* Successful compilation
* Provided test cases
* Additonal test cases
* Performance score

The point totals of each are listed in the forum announcement post.
