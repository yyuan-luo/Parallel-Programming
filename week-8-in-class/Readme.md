# In-class exercise 8: Help parallelize wave simulation
This time, we are going to face the wave simulation again! Remember last time that Jack Sparrow has successfully found his way to his treasure on a stromy sea? Since we want to extend the parallelization to multiple computers with distributed memory, we need the help of MPI library! This time, let's work on a simpler wave simulation problem, but with MPI. Let's see how much time you can save for us!


## Your tasks
1. Add Intel Intrinsics vector operations to **student_submission.cpp** to achieve parallelism.
2. Submit your code written in **student_submission.cpp** to [https://parprog.caps.in.tum.de/](https://parprog.caps.in.tum.de/).
3. Make sure you reach a speedup of 2.

Remember - you only have 10 minutes.

## Technical details
The files Utility.h and Makefile were already sent to the server. You can only modify student_submission.cpp (there are multiple TODOs comment in the code, finish them). As soon as you have optimized your code, upload ONLY student_submission.cpp to the server and check if your message was sent successfully.

Good luck! 

## How to run the code

```bash
make
# run sequential implementation
./sequential_implementation
# run your solution
mpirun ./student_implementation
```