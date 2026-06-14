cmake_minimum_required(VERSION 3.22...3.30)
project(exp_dram_loopback)

add_executable(exp_dram_loopback)
target_sources(exp_dram_loopback PRIVATE dram_loopback.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(exp_dram_loopback PUBLIC TT::Metalium)
