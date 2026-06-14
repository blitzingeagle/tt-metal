#include <fmt/ostream.h>
#include <cstdint>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

void matmul_single_core(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& c,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr CoreCoord core = {0, 0};
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // Number of tiles for each dimension
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // Allocate DRAM buffers for input and output data (replicated per device across the mesh).
    // Setting page_size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
    };
    distributed::ReplicatedBufferConfig buffer_config_A{.size = sizeof(bfloat16) * a.size()};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = sizeof(bfloat16) * b.size()};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = sizeof(bfloat16) * c.size()};
    std::shared_ptr<distributed::MeshBuffer> src0_dram_buffer =
        distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> src1_dram_buffer =
        distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> dst0_dram_buffer =
        distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());

    // Create circular buffers for input and output data.
    DataFormat cb_data_format = DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;

    uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig{
            num_input_tiles * single_tile_size,
            {{src0_cb_index, cb_data_format}},
        }
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig{
            num_input_tiles * single_tile_size,
            {{src1_cb_index, cb_data_format}},
        }
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t dst0_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_dst0_config =
        CircularBufferConfig{
            num_output_tiles * single_tile_size,
            {{dst0_cb_index, cb_data_format}},
        }
            .set_page_size(dst0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_dst0_config);

    // Create data movement kernels
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);  // TODO: Why no ->get_backing_buffer()
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "mm_single_core/kernels/dataflow/reader_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = RISCV_1_default,
            .compile_args = reader_compile_time_args,
        });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst0_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "mm_single_core/kernels/dataflow/writer_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = RISCV_0_default,
            .compile_args = writer_compile_time_args,
        });

    // Create compute kernel
    std::vector<uint32_t> compute_compile_time_args = {Mt, Kt, Nt};
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "mm_single_core/kernels/compute/mm_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .compile_args = compute_compile_time_args,
        });

    // Set kernel arguments
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst0_addr = dst0_dram_buffer->address();
    SetRuntimeArgs(program, reader_kernel_id, core, {src0_addr, src1_addr, Mt, Kt, Nt});
    SetRuntimeArgs(program, writer_kernel_id, core, {dst0_addr, Mt, Kt, Nt});

    // Upload input data to DRAM buffers and execute kernels
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, a, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::EnqueueReadMeshBuffer(cq, c, dst0_dram_buffer, true);
}

int main() {
    fmt::print("MATMUL\n");

    bool pass = true;

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        constexpr uint32_t M = 32;
        constexpr uint32_t N = 32;
        constexpr uint32_t K = 32;

        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
        static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");
        static_assert(K % TILE_WIDTH == 0, "K must be divisible by TILE_WIDTH");

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<bfloat16> src0_vec(M * K);
        std::vector<bfloat16> src1_vec(K * N);

        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

        for (int r = 0; r < M; r++) {
            for (int c = 0; c < K; c++) {
                fmt::print("{:.2f}\t", float(src0_vec[r * K + c]));
            }
            fmt::print("\n");
        }

        for (int r = 0; r < K; r++) {
            for (int c = 0; c < N; c++) {
                fmt::print("{:.2f}\t", float(src1_vec[r * N + c]));
            }
            fmt::print("\n");
        }

        // Convert input matrices to 32x32 blocks, reorder so each 32x32 is stored contiguously
        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        std::vector<bfloat16> dst0_vec(M * N, 0);
        matmul_single_core(src0_vec, src1_vec, dst0_vec, M, N, K, mesh_device);
        dst0_vec = untilize_nfaces(dst0_vec, M, N);

        for (int r = 0; r < M; r++) {
            for (int c = 0; c < N; c++) {
                fmt::print("{:.2f}\t", float(dst0_vec[r * N + c]));
            }
            fmt::print("\n");
        }

        fmt::print("Matmul result of size {}\n", dst0_vec.size());

        if (!mesh_device->close()) {
            pass = false;
        }
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
