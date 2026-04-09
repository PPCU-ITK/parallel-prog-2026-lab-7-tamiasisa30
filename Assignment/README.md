# Assignment

The present work extends the original CFD Euler solver by introducing GPU acceleration using OpenMP target offloading. GPU acceleration is implemented by offloading the computationally intensive loops to the device using OpenMP directives such as “#pragma omp target teams distribute parallel for”. A key modification is the use of parallel reduction on the GPU to compute the total kinetic energy, which allows efficient accumulation across threads. Performance comparisons were carried out for increasing grid sizes (Nx, Ny) scaled by factors of 1×, 4×, 8×, and 16×. The results show that while the CPU and GPU produce nearly or identical kinetic energy values, the GPU achieves significantly faster execution times for larger problem sizes, demonstrating the effectiveness of parallel acceleration.

## How to compile
module load nvhpc

module load cuda

module load craype-accel-nvidia80

nvc++ -mp=gpu -gpu=cc80 -Ofast cfd_euler.cpp -o cfd -Minfo=accel,mp

## How to run
srun -p gpu --gres=gpu:1 --ntasks=1 --time=00:05:00 --mem=40G ./cfd
