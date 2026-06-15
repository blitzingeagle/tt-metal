cmake_minimum_required(VERSION 3.22...3.30)
project(exp_mm_multi_core)

add_executable(exp_mm_multi_core)
target_sources(exp_mm_multi_core PRIVATE matmul.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(exp_mm_multi_core PUBLIC TT::Metalium)
