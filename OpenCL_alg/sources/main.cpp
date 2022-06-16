#include <iostream>
#include <vector>
#include <optional>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "CL/opencl.hpp"

std::optional<cl::Platform> get_platform() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto& p : platforms) {
        cl_uint numDevices = 0;
        clGetDeviceIDs(p(), CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        if (numDevices > 0) return cl::Platform(p);
    }
    return std::nullopt;
}

int main()
{
    const auto PlatformOpt = get_platform();
    if (!PlatformOpt.has_value()) {
        std::cout << "Can't get platform!";
        std::getchar();
        exit(1);
    }
    auto& Platform = PlatformOpt.value();

	auto name = Platform.getInfo<CL_PLATFORM_NAME>();
	auto profile = Platform.getInfo<CL_PLATFORM_PROFILE>();
    auto version = Platform.getInfo<CL_PLATFORM_VERSION>();
    auto vendor = Platform.getInfo<CL_PLATFORM_VENDOR>();

    std::cout << "Selected platform: " << name << "\nVersion: "<< version <<", Profile: " << profile << "\nVendor:  " << vendor << std::endl;

    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM,
      reinterpret_cast<cl_context_properties>(Platform()),
      0
    };

    cl::Context C(CL_DEVICE_TYPE_GPU, properties);

    std::getchar();
    return 0;
}