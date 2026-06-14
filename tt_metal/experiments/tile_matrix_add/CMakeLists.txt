cmake_minimum_required(VERSION 3.22...3.30)
project(exp_tile_matrix_add)

add_executable(exp_tile_matrix_add)
target_sources(exp_tile_matrix_add PRIVATE tile_matrix_add.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(exp_tile_matrix_add PUBLIC TT::Metalium)
