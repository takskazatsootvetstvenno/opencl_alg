cmake_minimum_required(VERSION 3.12)

SET(PROJECT_NAME OpenCL_ALG)

get_filename_component(GENERAL_FOLDER ../ ABSOLUTE)
set(CMAKE_INSTALL_PREFIX "${GENERAL_FOLDER}/install" CACHE PATH "${GENERAL_FOLDER}/install" FORCE)

project(${PROJECT_NAME})
                                                                        
option(OPENCL_SDK_BUILD_SAMPLES OFF)
option(BUILD_DOCS OFF)
option(BUILD_EXAMPLES OFF)
option(BUILD_TESTING OFF)
add_subdirectory(OpenCL_alg)

if(MSVC)
	 set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT OpenCL_ALG)
	 add_compile_options(/MP)
endif()