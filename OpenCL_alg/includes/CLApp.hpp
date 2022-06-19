#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "CL/opencl.hpp"
#include "CLApp.hpp"

namespace ALG
{
	class CLApp
	{
	public:
		CLApp(int local_x, int local_y);
		cl::Event vec_add(cl_int const* APtr, cl_int const* BPtr, cl_int* CPtr, size_t vec_size) noexcept;
		cl::Event mat_mult(const float* APtr, const float* BPtr, float* CPtr, int AX, int AY, int BY) noexcept;
		static constexpr int LOCAL_SIZE = 1;

		using vecAdd_t = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>;
		using matMult_t = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int>;
	private:
		cl::Platform get_platform() noexcept;
		cl::Context get_context(cl_platform_id p_id) noexcept;

		cl::Platform m_platform;
		cl::Context m_context;
		cl::CommandQueue m_queue;
		cl_context_properties m_properies;
		int m_local_x = 1;
		int m_local_y = 0;
		bool m_isValid = true;
	};
}