# Understanding GPU basics
As you're probably reading this, you're most likely aware that we're living in the age of the AI boom. Large Language Models like ChatGPT, Claude, Deepseek etc are widely used by a lot of users. However, have you ever sat down and asked yourself in which hardware these LLMs run on?

These LLMS run on specialized chips (processors) known as GPUs(Graphical Processing Units). In this blog I'm tasked with enabling you the reader to understand what a GPU is and how it differs from a CPU(Central Processing Unit).

Without further a do let's begin our quest of understanding what this GPU is.

## 1. GPU vs CPU
Think of a CPU as a master chef whose can make a variety of meals like ugali and cabbage, rice and stew, fried chicken etc and at the same time think of a GPU as thousands of chefs who can make the same meal like for example ugali and cabbage.

Let's go back in time from your high school life. I'm pretty sure majority of you in highschool had ugali and cabbage for lunch. Taking the role of a school principal would you hire one master chef or hire say around 20 chefs who cook ugali and cabbage quickly enough to meet the demand of the highschool students? The answer is pretty simple, you would hire those 20 chefs since they would be much quicker than that one master chef.

### CPU and GPU architecture
![output:](diagram.png)

- From out analogy from the previous section, we can think of chefs as being represented by the cores in the above diagram.
- These cores are responsible for reading and executing instructions. You can think of instructions as commands given to a computer to direct it to do a certain task.
- 
Say we have two vector of 100 elements each. Between the CPU and GPU, among the two which one do you think would be faster than the other? The GPU would be faster since we can clearly see that it has more cores than the CPU. A core is a physical hardware component responsible for the execution of instructions.

### Execution Process
- Let's have a scenario in which  our GPU had 100 cores and our CPU had 1 core, and we are tasked to add two vectors with 100 elements each.
- Let's name these two vectors A and B, their sum will be stored in C.
- For a our CPU code, we'll use an ordinary for loop shown below.

```cpp
void sequential_vector_addition(float*A,float*B,float*C,int N)
{
  for(int i=0;i<N;++i)
{
  C[i] = A[i] + B[i]
}
}

```
- From the above code we can see that this for-loop that 100 iterations meaning this loop will go on for 100 rounds so that we finally have the sum of the two vectors in C.

- Our GPU code will be written in a language known as CUDA and the code is shown below:
```cpp
__global__ void vector_addition(float *A,float*B ,float* C,int N)
{
  int index = threadIdx.x;
  if(index <N)
  {
    C[index] = A[index] + B[index];
  }
}
```
- From the code above, I've written a function known as a **kernel**. A kernel is a function is executed on the GPU, that why it has the keyword `__global__` indicating that it will be executed on the GPU. The arguments of the kernel are our vector A, B  and C and the length of our vector `N` which is 100.
-  There's also and index variable called and it's been assigned a value of `threadIdx.x` which means the coordinate of an individual thread along the `x` axis. Each thread will responsible for calculating the sum of one element.
- The if-statements makes sure that threads whose indices are less than 100 will be active in the sum calculation.
- The line `C[index] = A[index] + B[index]` means the threads o id 0 to 99 will perform the addittion.
- So we can see that the vector addition in a GPU only goes for one round since there are many threads which perform addition in parallel. Hence that's why it's much faster than the CPU since the has only 1 thread hence it requires 100 rounds to execute sequentially.
