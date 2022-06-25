#include <iostream>
#include <vector>
#include <optional>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <iomanip>
#include "CLApp.hpp"
#include "cpu_helper_functions.hpp"

int main()
{
    std::chrono::high_resolution_clock::time_point TimeStart, TimeFin;
    cl_ulong GPUTimeStart, GPUTimeFin;
    long Dur, GDur;

    ALG::CLApp app(1, 1);

    const int k = 2;
    const int AX = 3 * k; //num of row
    const int AY = 2 * k; //num of column
    const int BY = 1 * k;
    cl::vector<float> A(AX * AY), AT(AX * AY), B(AY * BY), C(AX * BY), C_CPU(AX * BY), C_CPU_OPTIMIZED(AX * BY);
    //AX - Num of rows,
    //AY - Num of columns
    CPU_HELPER_FUNC::rand_init(A.begin(), A.end(), 0, 10);
    CPU_HELPER_FUNC::rand_init(B.begin(), B.end(), 0, 10);

    std::cout << "C MAT SIZE: rows: " << AX << "  columns: " << BY << std::endl;

    {
        std::cout << "\n-mat_mult_GPU\n";
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
        std::cout << "GPU pure time measured: " << GDur << " ms" << std::endl;
    }
    {
        std::cout << "\n-multiply_CPU_simple\n";
        TimeStart = std::chrono::high_resolution_clock::now();
        CPU_HELPER_FUNC::multiply_CPU_simple(A.data(), B.data(), C_CPU.data(), AX, AY, BY);
        TimeFin = std::chrono::high_resolution_clock::now();
        Dur =
            std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart)
            .count();
        std::cout << "CPU time measured: " << Dur << " ms" << std::endl;
        auto transpose_test_pass = CPU_HELPER_FUNC::compare_matrices(C_CPU.data(), C.data(), AX, BY);
        if (transpose_test_pass)
            std::cout << "CPU and GPU multiplication are equel\n";
    }
    {
        std::cout << "\n-multiply_transpose_CPU\n";
        TimeStart = std::chrono::high_resolution_clock::now();
        CPU_HELPER_FUNC::multiply_transpose_CPU(A.data(), B.data(), C_CPU_OPTIMIZED.data(), AX, AY, BY);
        TimeFin = std::chrono::high_resolution_clock::now();
        Dur =
            std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart)
            .count();
        std::cout << "CPU_optimized time measured: " << Dur << " ms" << std::endl;
        auto transpose_test_pass = CPU_HELPER_FUNC::compare_matrices(C_CPU_OPTIMIZED.data(), C.data(), AX, BY);
        if (transpose_test_pass)
            std::cout << "CPU and GPU multiplication are equel\n";
    }
    {
        std::cout << "\n-mat_transpose_GPU\n";
        TimeStart = std::chrono::high_resolution_clock::now();
        auto evt_transpose = app.mat_transpose(A.data(), AT.data(), AX, AY);
        TimeFin = std::chrono::high_resolution_clock::now();
        Dur =
            std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart)
            .count();
        std::cout << "GPU transpose wall time measured: " << Dur << " ms" << std::endl;
        GPUTimeStart = evt_transpose.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        GPUTimeFin = evt_transpose.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        GDur = (GPUTimeFin - GPUTimeStart) / 1000000; // ns -> ms
        std::cout << "GPU transpose pure time measured: " << GDur << " ms" << std::endl;
    }
    {
        std::cout << "\n-transpose_CPU\n";
        cl::vector<float> AT_CPU(AX * AY);
        TimeStart = std::chrono::high_resolution_clock::now();
        CPU_HELPER_FUNC::transpose_CPU(A.data(), AT_CPU.data(),AX, AY);
        TimeFin = std::chrono::high_resolution_clock::now();
        Dur =
            std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart)
            .count();
        std::cout << "CPU time measured: " << Dur << " ms" << std::endl;
        auto transpose_test_pass = CPU_HELPER_FUNC::compare_matrices(AT.data(), AT_CPU.data(), AX, AY);
        if (transpose_test_pass)
            std::cout << "CPU and GPU transpose are equel\n";
    }
    if (AX * BY < 50)
    {
        std::cout << "--- Matrix A---\n";
        CPU_HELPER_FUNC::print_matrix(A.data(), AX, AY);
        std::cout << "--- Matrix B---\n";
        CPU_HELPER_FUNC::print_matrix(B.data(), AY, BY);
        std::cout << "--- Matrix C mult GPU---\n";
        CPU_HELPER_FUNC::print_matrix(C.data(), AX, BY);
        /*std::cout << "--- Matrix C mult CPU---\n";
        CPU_HELPER_FUNC::print_matrix(C_CPU.data(), AX, BY);
        std::cout << "--- Matrix C mult CPU OPTIMIZED---\n";
        CPU_HELPER_FUNC::print_matrix(C_CPU_OPTIMIZED.data(), AX, BY);
        */
        std::cout << "--- Matrix AT traspose GPU---\n";
        CPU_HELPER_FUNC::print_matrix(AT.data(), AY, AX);
        std::cout << "--- End Matrices---\n";
    }
    if (app.isValid())
        std::cout << "Success!";
    else
        std::cout << "smth wrong!";
    std::getchar();
    return 0;
}