#pragma once
namespace CPU_HELPER_FUNC
{
    template <typename It>
    void rand_init(It start, It end, const float low, const float up) {
        static std::mt19937_64 mt_source;
        std::uniform_int_distribution<int> dist(low, up);
        for (It cur = start; cur != end; ++cur)
            *cur = dist(mt_source);
    }

    template <typename T>
    void print_matrix(const T* M, const int MX, const int MY) {
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
        for (i = 0; i < Ax; ++i)
            for (j = 0; j < By; ++j)
            {
                T sum = 0;
                for (r = 0; r < Ay; ++r)
                    sum += A[Ay * i + r] * B[By * r + j];
                C[By * i + j] = sum;
            }
    }

    template<typename T>
    void transpose_CPU(const T* A, T* AT, const size_t Ax, const size_t Ay) noexcept
    {
        size_t i, j;
        for (i = 0; i < Ax; ++i)
            for (j = 0; j < Ay; ++j)
                AT[j * Ax + i] = A[i * Ay + j];
    }

    template<typename T>
    void multiply_transpose_CPU(const T* A, const T* B, T* C, const size_t Ax, const size_t Ay, const size_t By) noexcept
    {
        std::vector<T> tmp(By * Ay);
        size_t i, j, r;

        for (i = 0; i < Ay; ++i)
            for (j = 0; j < By; ++j)
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

    template<typename T>
    [[nodiscard]] bool compare_matrices(const T* M1, const T* M2, const size_t Ax, const size_t Ay)
    {
        for (size_t i = 0; i < Ax * Ay; ++i)
            if (M1[i] != M2[i]) {
                std::cerr << "Error while compare!\ni = " << i / Ax << " | j = " << i % Ay << std::endl;
                return false;
            }
        return true;
    }
}