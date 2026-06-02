cmake_minimum_required(VERSION 3.22...3.30)
project(exp_hello_world)

add_executable(exp_hello_world)
target_sources(exp_hello_world PRIVATE hello_world.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(exp_hello_world PUBLIC TT::Metalium)
