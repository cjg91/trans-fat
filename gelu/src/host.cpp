/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <ctime>

#define NUM_TESTS 100
#define TEST_LEN 2048

float swGELU(const float x)
{

  float th = sqrtf(2.0 / M_PI) * (x + X3_CONST * powf(x, 3));
  float gx = 0.5 * x * (1 + tanhf(th));

  return gx;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string binaryFile = argv[1];
  size_t vector_size_bytes = sizeof(float) * TEST_LEN;
  cl_int err;
  cl::Context context;
  cl::Kernel krnl_vector_add;
  cl::CommandQueue q;
  // Allocate Memory in Host Memory
  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
  // hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice
  // but to create
  // its own host side buffer. So it is recommended to use this allocator if
  // user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
  // boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with
  // CL_MEM_USE_HOST_PTR
  std::vector<float, aligned_allocator<float>> source_in1(TEST_LEN);
  std::vector<float, aligned_allocator<float>> source_hw_results(TEST_LEN);
  std::vector<float, aligned_allocator<float>> source_sw_results(TEST_LEN);

  // Create the test data
  srand(std::time());
  std::generate(source_in1.begin(), source_in1.end(), std::rand);
  for (int i = 0; i < TEST_LEN; i++) {
    source_sw_results[i] = swGELU(source_in1);
    source_hw_results[i] = 0;
  }

  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  auto devices = xcl::get_xil_devices();
  // read_binary_file() is a utility API which will load the binaryFile
  // and will return the pointer to file buffer.
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  int valid_device = 0;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err, krnl_gelu_driver = cl::Kernel(program, "gelu", &err));
      valid_device++;
      break; // we break because we found a valid device
    }
  }
  if (valid_device == 0) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  OCL_CHECK(err, cl::Buffer buffer_in1(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     vector_size_bytes, source_in1.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_output(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                     vector_size_bytes, source_hw_results.data(), &err));

  //int size = DATA_SIZE;
  OCL_CHECK(err, err = krnl_gelu_driver.setArg(0, buffer_in1));
  OCL_CHECK(err, err = krnl_gelu_driver.setArg(1, buffer_output));
  //OCL_CHECK(err, err = krnl_vector_add.setArg(3, size));

  double total_time = 0;
  for (int i = 0; i < NUM_TESTS; ++i)
  {
    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1},
                    0 /* 0 means from host*/));
    
    q.finish();

    //start timer
    double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);
    auto kernel_start = std::chrono::high_resolution_clock::now();

    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is
    // recommended
    // to always use enqueueTask() for invoking HLS kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_gelu_driver));

    q.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},
                                                    CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    // OPENCL HOST CODE AREA END
    
    // print timing results
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    kernel_time_in_sec = kernel_time.count();
    total_time += kernel_time_in_sec;
  }  

  std::cout<<"Ran "<<NUM_TESTS<<" Tests"<<std::endl;
  std::cout<<"Exectuion Time: "<<total_time/NUM_TESTS<<std::endl;

  // Compare the results of the Device to the simulation
  bool match = true;
  for (int i = 0; i < TEST_LEN; i++) {
    if (source_hw_results[i] != source_sw_results[i]) {
      std::cout << "Error: Result mismatch" << std::endl;
      std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                << " Device result = " << source_hw_results[i] << std::endl;
      match = false;
      break;
    }
  }

  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
