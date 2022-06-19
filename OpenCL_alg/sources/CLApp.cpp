#include "CLApp.hpp"
#include <iostream>
#include <chrono>

const char* vecAddKernelSRC = R"(

__kernel void vector_add(__global int* A, __global int* B, __global int* C) {
	int i = get_global_id(0);
	C[i] = A[i] + B[i];
}

)";
const char* matMultKernelSRC = R"(

#define TYPE float
__kernel void matrix_multiply(__global TYPE* A, __global TYPE* B, __global TYPE* C, int AX, int AY, int BY) {
	int row = get_global_id(0);
	int col = get_global_id(1);
	TYPE sum = 0;

	for (int k = 0; k < AY; ++k)
		sum += A[row * AY + k] * B[k * BY + col];

	C[row * BY + col] = sum;
}

)";

namespace ALG
{
	cl::QueueProperties getQueueProp() noexcept
	{
		return cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder;
	}

	CLApp::CLApp(int local_x, int local_y)
		:m_platform(get_platform()),
		m_context(get_context(m_platform())),
		m_queue(m_context, getQueueProp()),
		m_local_x(local_x),
		m_local_y(local_y)
	{
		auto name = m_platform.getInfo<CL_PLATFORM_NAME>();
		auto profile = m_platform.getInfo<CL_PLATFORM_PROFILE>();
		auto version = m_platform.getInfo<CL_PLATFORM_VERSION>();
		auto vendor = m_platform.getInfo<CL_PLATFORM_VENDOR>();

		std::cout << "Selected platform: " << name << "\nVersion: " 
			<< version << ", Profile: " << profile << "\nVendor:  "
			<< vendor << std::endl << std::endl;
	}

	cl::Event CLApp::vec_add(cl_int const* APtr, cl_int const* BPtr, cl_int* CPtr, size_t vec_size) noexcept
	{
		size_t BufSz = vec_size * sizeof(cl_int);

		cl::Buffer A(m_context, CL_MEM_READ_ONLY, BufSz);
		cl::Buffer B(m_context, CL_MEM_READ_ONLY, BufSz);
		cl::Buffer C(m_context, CL_MEM_WRITE_ONLY, BufSz);

		cl::copy(m_queue, APtr, APtr + vec_size, A);
		cl::copy(m_queue, BPtr, BPtr + vec_size, B);

		cl::Program program(m_context, vecAddKernelSRC);
		cl_int buildErr = program.build();
		if (buildErr != CL_SUCCESS) {
			std::cerr << "Kernel building error: " << buildErr << "\n";
			m_isValid = false;
		}
		vecAdd_t add_vecs(program, "vector_add");

		cl::NDRange GlobalRange(vec_size);
		cl::NDRange LocalRange(LOCAL_SIZE);
		cl::EnqueueArgs Args(m_queue, GlobalRange, LocalRange);
		
		cl::Event evt = add_vecs(Args, A, B, C);
		auto time_start = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		evt.wait();
		cl::copy(m_queue, C, CPtr, CPtr + vec_size);
		auto time_end = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		std::cout << "Time on GPU side: " << time_end - time_start << std::endl;
		return evt;
	}

	cl::Event CLApp::mat_mult(const float* APtr, const float* BPtr, float* CPtr,
		int AX, int AY, int BY) noexcept{
		size_t ASz = AX * AY, ABufSz = ASz * sizeof(float);
		size_t BSz = AY * BY, BBufSz = BSz * sizeof(float);
		size_t CSz = AX * BY, CBufSz = CSz * sizeof(float);

		cl::Buffer A(m_context, CL_MEM_READ_ONLY, ABufSz);
		cl::Buffer B(m_context, CL_MEM_READ_ONLY, BBufSz);
		cl::Buffer C(m_context, CL_MEM_WRITE_ONLY, CBufSz);

		cl::copy(m_queue, APtr, APtr + ASz, A);
		cl::copy(m_queue, BPtr, BPtr + BSz, B);

		cl::Program program(m_context, matMultKernelSRC);
		cl_int buildErr = program.build();
		if (buildErr != CL_SUCCESS) {
			std::cerr << "Kernel building error: " << buildErr << "\n";
			m_isValid = false;
		}

		matMult_t gemm(program, "matrix_multiply");

		cl::NDRange GlobalRange(AX, BY);
		cl::NDRange LocalRange(m_local_x, m_local_y);
		cl::EnqueueArgs Args(m_queue, GlobalRange, LocalRange);

		cl::Event Evt = gemm(Args, A, B, C, AX, AY, BY);
		Evt.wait();

		cl::copy(m_queue, C, CPtr, CPtr + CSz);
		return Evt;
	}

	cl::Platform CLApp::get_platform() noexcept
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		for (auto& p : platforms) {
			cl_uint numDevices = 0;
			clGetDeviceIDs(p(), CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
			if (numDevices > 0)
				return cl::Platform(p);
		}
		m_isValid = false;
	}

	cl::Context CLApp::get_context(cl_platform_id p_id) noexcept
	{
		cl_context_properties properties[] = {
			CL_CONTEXT_PLATFORM,
			reinterpret_cast<cl_context_properties>(p_id),
			0
		};
		return cl::Context(CL_DEVICE_TYPE_GPU, properties);
	}
}