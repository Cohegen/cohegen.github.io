# Unpacking the CUDA Coordinate Mapping

![out:](assets/ralph_wiggum.png)

I'm sure all of you have watched *The Simpsons*, probably during your childhood. In the picture above, we can see a character known as **Ralph Wiggum**. In this scene, he utters some well-known words: "I'm in danger."

However, in our case, the caption reads, "Where am I?"

Let's picture Ralph as an individual thread whose sole task is to add a pair of elements from vectors A and B and write the result into vector C.

- In the previous blog, ["GPU Basics"](https://cohegen.github.io/blogs/gpu_basics.html), there was a line of code that looked like this:

```cpp
int index = threadIdx.x;
```

- In this blog, we'll embark on a quest to understand what that line of code really means.

Now, back to our dear friend Ralph. We've decided to picture him as an individual thread. However, Ralph is on a bus, as we can clearly see. And that bus is expected to have many students on board.

These other students could be , for example, other characters like Bart, Lisa, or even Milhouse. Each student represents an individual thread, which means the bus itself represents something known as a **Thread Block**, or simply a **block**.

- To put it simply, we can think of a block as a container or space that holds multiple threads.

# Mapping Individual Threads

- Let's go back in time to our primary school years. During school trips, if you had paid for the trip, you would be assigned a bus to travel in, as well as a seat number on that bus.

- Let's say your school had hired around five buses, and each bus could accommodate approximately 30 students.

- In case of a disciplinary issue involving a particular student during the trip, the teacher in charge would first need to know which bus the student was on and what their seat number was.

- Since there were five buses, let's assign each bus a unique number: `0, 1, 2, 3, 4`.

- We can equate these numbers with a CUDA variable known as `blockIdx`. `blockIdx` represents the bus number.

- Each bus has 30 students in total. We can equate the total capacity of each bus with the CUDA variable `blockDim`. `blockDim` simply answers the question: How many threads or students does each bus/block have?

- Since every student on each bus has a seat number, we can assign that number to `threadIdx`.

- However, let's say we wanted to retrieve some general information about a student, such as their unique number. Suppose a particular student has the following information:

```cpp
bus_number/blockIdx.x = 3
seat_number/threadIdx.x = 20
```

- Since `blockDim.x = 30`, we can determine the student's unique number using the formula below:

```cpp
unique_number = bus_number * blockDim.x + seat_number
```

Which translates into:

```cpp
unique_number = blockIdx.x * blockDim.x + threadIdx.x
unique_number = 3 * 30 + 20 = 110
```

- So, our student's unique number is `110`.

# Mapping into 1-D Arrays

In the kernel from the previous blog, we mapped our index as follows:

```cpp
int index = threadIdx.x;
```

- Note that we aren't using `blockIdx` or `blockDim` here. We were using only one thread block, and we assumed that it had 100 threads capable of performing parallel vector addition on vectors A and B and writing the results into C.

- However, what if we were performing vector addition on vectors A and B, each containing 1,000 elements? Our individual thread block would only perform the addition of the first 100 elements, leaving the remaining 900 unattended.

  However, there's an optimization we can introduce to make our thread block capable of handling the addition of the two vectors. We'll explore this later.

- So, we have two vectors, each with 1,000 elements, but initially, we have only one thread block containing 100 threads. Since our GPU has adequate resources, we can invoke nine additional thread blocks, each containing 100 threads, to join the bandwagon and perform the addition.

  These thread blocks collectively form what is known as a **Grid**. A grid is simply a collection of thread blocks launched by a kernel.
  This can be reprensented in the diagram belo:

  ![out:](block.png)

- Since vectors are 1-D (one-dimensional) arrays, we would have the following identifiers: `blockIdx.x`, `blockDim.x`, and `threadIdx.x`. They have the suffix `.x`, representing the X dimension of CUDA's indexing system.

- Now we have 10 blocks. The indices of these blocks follow the indexing convention used in programming, meaning that our first index is always `0`, while the last index is `N-1`. In our case, that would be `10-1 = 9`. The same also applies to threads.

- Given a thread with an index (`threadIdx.x`) of `5` in a thread block with an index (`blockIdx.x`) of `6`, we want to determine which element in vectors A and B that thread will be responsible for processing.

  To find this, we use the formula below:

```cpp
index = blockIdx.x * blockDim.x + threadIdx.x
index = 6 * 100 + 5 = 605
```

- This means that **Thread 5 in Block 6** will be responsible for calculating the sum of `A[605] + B[605]` and writing the result into `C[605]`.

- Thus, for vectors A and B with 1,000 elements each, our vector addition kernel evolves into:

```cpp
__global__ void vector_addition(float* A, float* B, float* C, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
    {
        C[index] = A[index] + B[index];
    }
}
```

- Here, `N = 1000`.

- The boundary check `if (index < N)` ensures that threads whose calculated indices exceed the size of our vectors do not attempt to access memory outside their allocated bounds.
