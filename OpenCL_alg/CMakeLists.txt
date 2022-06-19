cmake_minimum_required(VERSION 3.12)

set(ALG_PROJECT_NAME OpenCL_ALG)
                  
set(ALG_INCLUDES
	includes/CLAPP.hpp
)

set(ALG_SOURCES
	sources/main.cpp
	sources/CLApp.cpp
)
add_executable(${ALG_PROJECT_NAME}
	${ALG_INCLUDES}
	${ALG_SOURCES}
)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT ${ALG_PROJECT_NAME} )
target_compile_features(${ALG_PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${ALG_PROJECT_NAME} PUBLIC includes)
target_include_directories(${ALG_PROJECT_NAME} PRIVATE src)
                                                           
add_subdirectory(../external/OpenCL-SDK ${CMAKE_CURRENT_BINARY_DIR}/OpenCL-SDK)
target_link_libraries(${ALG_PROJECT_NAME} PRIVATE OpenCL::HeadersCpp)
target_link_libraries(${ALG_PROJECT_NAME} PRIVATE OpenCL)

install(TARGETS ${ALG_PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})