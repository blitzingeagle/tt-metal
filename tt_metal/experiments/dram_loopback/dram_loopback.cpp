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
    fmt::print("DRAM Loopback\n");

    bool pass = true;

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        constexpr uint32_t num_tiles = 50;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;  // 32x32
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;                      // 32x32x2
        constexpr uint32_t dram_buffer_size = tile_size_bytes * num_tiles;

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::DRAM,
        };
        distributed::DeviceLocalBufferConfig l1_config{
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::L1,
        };
        distributed::ReplicatedBufferConfig dram_buffer_config{
            .size = dram_buffer_size,
        };
        distributed::ReplicatedBufferConfig l1_buffer_config{
            .size = tile_size_bytes,
        };

        std::shared_ptr<distributed::MeshBuffer> l1_buffer =
            distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());
        std::shared_ptr<distributed::MeshBuffer> input_dram_buffer =
            distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
        std::shared_ptr<distributed::MeshBuffer> output_dram_buffer =
            distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        std::vector<uint32_t> dram_copy_compile_time_args;
        TensorAccessorArgs(*input_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);
        TensorAccessorArgs(*output_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);
        KernelHandle dram_loopback_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "dram_loopback/kernels/loopback_kernel.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = RISCV_0_default,
                .compile_args = dram_copy_compile_time_args,
            });

        std::vector<bfloat16> input_vec(elements_per_tile * num_tiles);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 100.f);
        for (auto& val : input_vec) {
            val = bfloat16(distribution(rng));
        }
        distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, false);

        const std::vector<uint32_t> runtime_args = {
            l1_buffer->address(),
            input_dram_buffer->address(),
            output_dram_buffer->address(),
            num_tiles,
        };
        SetRuntimeArgs(program, dram_loopback_kernel_id, core, runtime_args);
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);

        distributed::Finish(cq);

        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, output_dram_buffer, true);

        TT_FATAL(
            result_vec.size() == input_vec.size(),
            "Result vector size {} does not match input vector size {}",
            result_vec.size(),
            input_vec.size());
        for (int i = 0; i < input_vec.size(); i++) {
            if (input_vec[i] != result_vec[i]) {
                pass = false;
                break;
            }
        }

        if (!mesh_device->close()) {
            pass = false;
        }
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed!\n");
    } else {
        TT_THROW("Test Failed!");
    }

    return 0;
}
