#include <iostream>
#include <vector>
#include <optional>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <iomanip>
#include "CLApp.hpp"

template <typename It>
 void rand_init(It start, It end, float low, float up) {
    static std::mt19937_64 mt_source;
    std::uniform_int_distribution<int> dist(low, up);
    for (It cur = start; cur != end; ++cur)
        *cur = dist(mt_source);
}

 template <typename T>
 void outm(const T* M, int MX, int MY) {
     for (int i = 0; i < MX; ++i) {
         for (int j = 0; j < MY; ++j)
             std::cout << std::setw(3) << M[i * MY + j] << " ";
         std::cout << std::endl;
     }
 }

 template<typename T>
 void multiply_CPU_simple(const T* A, const T* B, T* C, const size_t Ax, const size_t Ay, const size_t By) noexcept
 {
     size_t i, j, r;
     for(i = 0; i < Ax; ++i)
         for (j = 0; j < By; ++j)
         {
             T sum = 0;
             for (r = 0; r < Ay; ++r) 
                 sum += A[Ay * i + r] * B[By * r + j];
             C[By * i + j] = sum;
         }
 }

 template<typename T>
 void multiply_transpose_CPU(const T* A, const T* B, T* C, const size_t Ax, const size_t Ay, const size_t By) noexcept
 {
     std::vector<T> tmp(By * Ay);
     size_t i, j, r;

     for (i = 0; i < Ay; ++i)
         for ( j = 0; j < By; ++j)
             tmp[Ay * j + i] = B[By * i + j];

     for (i = 0; i < Ax; ++i)
         for (j = 0; j < By; ++j)
         {
             T sum = 0;
             for (r = 0; r < Ay; ++r) 
                 sum += A[Ay * i + r] * tmp[Ay * j + r];
             C[By * i + j] = sum;
         }
 }

int main()
{
    std::chrono::high_resolution_clock::time_point TimeStart, TimeFin;
    cl_ulong GPUTimeStart, GPUTimeFin;
    long Dur, GDur;

    ALG::CLApp app(1, 1);

    const int k = 1100;
    int AX = 3*k; //num of row
    int AY = 2*k; //num of column
    int BY = 1*k;
    cl::vector<float> A(AX * AY), B(AY * BY), C(AX * BY), C_CPU(AX * BY), C_CPU_OPTIMIZED(AX * BY);

    rand_init(A.begin(), A.end(), 0, 10);
    rand_init(B.begin(), B.end(), 0, 10);
   
    std::cout << "C MAT SIZE: rows: " << AX << "  columns: " << BY <<std::endl;

    TimeStart = std::chrono::high_resolution_clock::now();

    cl::Event evt = app.mat_mult(A.data(), B.data(), C.data(), AX, AY, BY);

    TimeFin = std::chrono::high_resolution_clock::now();
    Dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart)
        .count();
    std::cout << "GPU wall time measured: " << Dur << " ms" << std::endl;
    GPUTimeStart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    GPUTimeFin = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    GDur = (GPUTimeFin - GPUTimeStart) / 1000000; // ns -> ms
    std::cout << "GPU pure time measured: " << GDur << " ns" << std::endl;
   
    TimeStart = std::chrono::high_resolution_clock::now();
    multiply_CPU_simple(A.data(), B.data(), C_CPU.data(), AX, AY, BY);
    TimeFin = std::chrono::high_resolution_clock::now();
    Dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart)
        .count();
    std::cout << "CPU time measured: " << Dur << " ms" << std::endl;

    TimeStart = std::chrono::high_resolution_clock::now();
    multiply_transpose_CPU(A.data(), B.data(), C_CPU_OPTIMIZED.data(), AX, AY, BY);
    TimeFin = std::chrono::high_resolution_clock::now();
    Dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart)
        .count();
    std::cout << "CPU_optimized time measured: " << Dur << " ms" << std::endl;
    
    if (AX * BY < 50)
    {
        std::cout << "--- Matrix A---\n";
        outm(A.data(), AX, AY);
        std::cout << "--- Matrix B---\n";
        outm(B.data(), AY, BY);
        std::cout << "--- Matrix C GPU---\n";
        outm(C.data(), AX, BY);
        std::cout << "--- Matrix C CPU---\n";
        outm(C_CPU.data(), AX, BY);
        std::cout << "--- Matrix C CPU OPTIMIZED---\n";
        outm(C_CPU_OPTIMIZED.data(), AX, BY);
        std::cout << "--- End Matrices---\n";
    }


    std::cout << "Success!";
    std::getchar();
    return 0;
}