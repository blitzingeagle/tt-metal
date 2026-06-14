#include <fmt/ostream.h>
#include <cstdint>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    fmt::print("Hello world!\n");

    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    KernelHandle hello_world_kernel_id = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "hello_world/kernels/hello_world_kernel.cpp", core, ComputeConfig{});

    SetRuntimeArgs(program, hello_world_kernel_id, core, {});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    fmt::print("Hello Core (0, 0) on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

    distributed::Finish(cq);
    fmt::print("Thank you, Core (0, 0) on Device 0, for the completed task.\n");
    mesh_device->close();

    return 0;
}
