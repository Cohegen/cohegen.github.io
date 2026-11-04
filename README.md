#  Unpacking the CUDA coordinate mapping

![out:](assets/ralph_wiggum.png)


I'm sure all of you have watched the simpson, probably from your childhood. In the picture above we can see a character known as **Ralph Wiggum**. In this scene, he utters some well known words, that is "I'm in danger".
However, in our case here, the caption is "where am I?". 
Let's picture Ralph as an individual thread who sole task is to add a pair of elements from vector A and B, and write the result into vector C.

- From the previous blog "https://cohegen.github.io/blogs/gpu_basics.html", there was a line of code which looked like this:
```
int index = threadIdx.x;
```
- In this blog we'll embark on quest to understand what that line of really means.

- Now back to our dear friend Ralph, we've decided to picture him as an individual thread. However, Ralph is in a bus, we can clearly see that. And the bus is expected to have many students in it. These other students for example can be other characters like Bart,Lisa or even Milhouse. So all students will individual threads and this means the bus will be something known as a *ThreadBlock* as simply a *block*.

- To put it in simple terms, we can think of a block as a sort of a container or space which contains threads.


# Mapping individual threads
- Let's go back in time, to our primary school years. During school trips, if you had paid for it, you were assigned the bus which you were to travel in and the seat number in the buxs.
- Let's say your school had hired around 5 buses and each bus could accommodate say 30 students each.
- Incase of a indiscipline case concerning a particular during the trip, the teacher in charge would need to first known which bus the student was in and what was his seat number
- Since there as five buses, let's assign these buses unique numbers say `0,1,2,3,4`.
- We can equate these numbers with a CUDA keyword known `blockIdx`. `blockIdx` represents the bus number.
- The buses have 30 students in totals each. We can equate total capacity of each bus with the keywod `blockDim`. `blockDim` simple answers the question how many threads/students does each bus/block have?
- And since every student in each bus a seat number we can assign it to `threadIdx`.
- However say we wanted to get the general information about a student say his/her unique number, say a particular student has the following information
```
bus_number/blockIdx =3
seat_number/threadIdx = 20
```
- Since `blockDim=5` we can map the student's unique number with the formular below:
```
unique_number = bus_number*num_buses + seat_number

```
Which translates into:
```
unique_number = blockIdx*blockDim + threadIdx.x
unique_number = 3*5 + 20 = 35 
```
- So our students unique number is `35`.

## Mapping into 1-d Arrays
In the kernel of the previous blog we mapped our index as:
```
int index = threadIdx.x
```
- Note that there's no use of `blockIdx` and `blockDim`, here we were using only one thread block and we assumed that it had 100 threads which could perform the parallel-wise vector addition of vectors A and B and write the result into C.
- However, what if we were performing vector addition between vector A and B each with 1000 elements. Our individual threadblock would only perform addition of the first 100 elements leaving the remaining 900 unattended. However there's an optimization we can do to make our threadblock be capable on performing addition between the two vectors,but this will be introduced later.

- So we have 2 vectors each of length 1000, but we have one threadblock with 100 threads, since our GPU has adequate resources we can invoke other 9 threadBlocks each with 100 threads to join the bandwagon of performing the addition.
- Since vector are 1-D (one dimensional) arrays we would have the following identifiers `blockIdx.x` `blockDim.x` and `threadIdx.x`. There have the  suffix `.x` implying our threads and blocks are along the X-axis.
- Say we have a thread in
