#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    DPRINT_MATH("Hello, I am the MATH core running the compute kernel.\n");
    DPRINT_UNPACK("Hello, I am the UNPACK core running the compute kernel.\n");
    DPRINT_PACK("Hello, I am the PACK core running the compute kernel.\n");
}
