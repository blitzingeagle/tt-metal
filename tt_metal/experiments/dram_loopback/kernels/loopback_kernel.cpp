#include <cstdint>

void kernel_main() {
    // Read the runtime arguments
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(2);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(3);

    // Get DRAM addresses for reading and writing data
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr);

    // Iterate the number of tiles, write from DRAM src to SRAM, then from SRAM to DRAM dst
    for (uint32_t i = 0; i < num_tiles; i++) {
        noc_async_read_page(i, in0, l1_buffer_addr);
        noc_async_read_barrier();  // Wait for read complete
        DPRINT("Copying Tile {} of {}.\n", i + 1, num_tiles);
        noc_async_write_page(i, out0, l1_buffer_addr);
        noc_async_write_barrier();  // Wait for write complete
    }
}
