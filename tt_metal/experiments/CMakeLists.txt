# All programming example targets should link tt_stl
link_libraries(TT::STL)

# Suppress narrowing warnings for experiments (being cleaned up in tt_metal first)
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")

add_compile_definitions(OVERRIDE_KERNEL_PREFIX="tt_metal/experiments/")
include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/experiments")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hello_world)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dram_loopback)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mm_single_core)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tile_matrix_add)

install(
    DIRECTORY
        hello_world
        dram_loopback
        mm_single_core
        tile_matrix_add
    # DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples
    # FIXME(afuller): Something funky is happening when installing files into
    # /usr/share/doc on a default Docker image. Speculation: some dependency for
    # magic doc handling is missing. For now keep it out of 'doc'.
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tt-metalium/examples"
    COMPONENT metalium-examples
    REGEX
    "/kernels/"
        EXCLUDE
)

install(
    DIRECTORY
        hello_world
        dram_loopback
        mm_single_core
        tile_matrix_add
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tenstorrent/kernels"
    COMPONENT metalium-examples
    FILES_MATCHING
    REGEX
    "/kernels/"
)
