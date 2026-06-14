//
// Created by morris on 02/06/26.
//
#include <fmt/ostream.h>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    fmt::print("TT_MATRIX_ADD\n");

    // constexpr int device_id = 0;
    // std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    // distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    //
    // distributed::MeshWorkload workload;
    // distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    //
    // Program program = CreateProgram();
}
