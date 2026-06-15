#include <fmt/ostream.h>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

void matmul_multi_core(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& c,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Check if the configuration is valid - matrices must be divisible by tile dimensions
    TT_ASSERT(
        (M * N) % TILE_HW == 0,
        "Matrix dimensions M={} and N={} must be divisible by TILE_HW={} to use this matmul implementation",
        M,
        N,
        TILE_HW);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    CoreCoord core_grid = mesh_device->compute_with_storage_grid_size();
    uint32_t num_output_tiles_total = (M * N) / TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(core_grid, num_output_tiles_total);

    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Kt = K / TILE_WIDTH;
    const uint32_t Nt = N / TILE_WIDTH;

    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // Allocate DRAM buffers
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
    };
    distributed::ReplicatedBufferConfig buffer_config_A{.size = single_tile_size * Mt * Kt};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = single_tile_size * Kt * Nt};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = single_tile_size * Mt * Nt};
    std::shared_ptr<distributed::MeshBuffer> src0_dram_buffer =
        distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> src1_dram_buffer =
        distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> dst0_dram_buffer =
        distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());

    // Create Circular Buffers
    DataFormat cb_data_format = DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;

    uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig{
            num_input_tiles * single_tile_size,
            {{src0_cb_index, cb_data_format}},
        }
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig{
            num_input_tiles * single_tile_size,
            {{src1_cb_index, cb_data_format}},
        }
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t dst0_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_dst0_config =
        CircularBufferConfig{
            num_input_tiles * single_tile_size,
            {{dst0_cb_index, cb_data_format}},
        }
            .set_page_size(dst0_cb_index, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_dst0_config);

    // Create Kernels
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "mm_multi_core/dataflow/reader_mm_output_tiles_partitioned.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = RISCV_1_default,
            .compile_args = reader_compile_time_args,
        });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst0_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "mm_multi_core/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = RISCV_0_default,
            .compile_args = writer_compile_time_args,
        });

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "mm_multi_core/compute/mm.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {},
        });

    uint32_t work_offset = 0;
    auto work_groups = {
        std::pair{core_group_1, work_per_core1},
        std::pair{core_group_2, work_per_core2},
    };

    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                SetRuntimeArgs(
                    program,
                    reader_kernel_id,
                    core,
                    {
                        src0_dram_buffer->address(),
                        src1_dram_buffer->address(),
                        Mt,
                        Kt,
                        Nt,
                        work_offset,
                        work_per_core,
                    });
                SetRuntimeArgs(
                    program,
                    writer_kernel_id,
                    core,
                    {
                        dst0_dram_buffer->address(),
                        work_per_core,
                        work_offset,
                    });
                SetRuntimeArgs(
                    program,
                    compute_kernel_id,
                    core,
                    {
                        work_per_core,
                        Kt,
                    });
                work_offset += work_per_core;
            }
        }
    }

    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::EnqueueReadMeshBuffer(cq, c, dst0_dram_buffer, true);
}

int main() {
    fmt::print("MATMUL MULTI CORE\n");

    bool pass = true;
    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        constexpr uint32_t M = 640;
        constexpr uint32_t N = 640;
        constexpr uint32_t K = 640;

        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
        static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");
        static_assert(K % TILE_WIDTH == 0, "K must be divisible by TILE_WIDTH");

        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Nt = N / TILE_WIDTH;
        constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<bfloat16> src0_vec(M * K);
        std::vector<bfloat16> src1_vec(K * N);

        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        std::vector<bfloat16> dst0_vec(dram_buffer_C_size / sizeof(bfloat16));
        matmul_multi_core(src0_vec, src1_vec, dst0_vec, M, N, K, mesh_device);
        dst0_vec = untilize_nfaces(dst0_vec, M, N);

        fmt::print("Matmul result of size {}\n", dst0_vec.size());

        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
    }

    if (pass) {
        fmt::print("Test Passed!\n");
    } else {
        TT_THROW("Test Failed!");
    }

    TT_ASSERT(pass);

    return 0;
}
