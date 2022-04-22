#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include "pipeline.hpp"
#include "config.hpp"
#include <stdlib.h>
#include <time.h>

void printmat(int8_t *A, const int M, const int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << int(A[i * N + j]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void genmat(T *A, const int M, const int N, const int mod)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = (i * N + j) % mod;
        }
    }
}

template <typename T>
const bool check(T *A, T *B, const int M, const int N)
{
    for (int i = 0; i < M * N; i++)
    {
        if (A[i] != B[i])
            return false;
    }
    return true;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}

	std::string binaryFile = argv[1];
	cl_int err;
	cl::Context context;
	cl::Kernel krnl;
	cl::CommandQueue q;

    /********** STAGE 3 ARGS ***********/
	std::vector<int8_t, aligned_allocator<int8_t>> stage3_fc_in(CFG::seqlen*CFG::dmodel);
	std::vector<int8_t, aligned_allocator<int8_t>> stage3_dense_weight_t(CFG::dmodel*CFG::ffdim);
	std::vector<int32_t, aligned_allocator<int32_t>> stage3_dense_bias(CFG::ffdim);
	std::vector<int8_t, aligned_allocator<int8_t>> fc3_to_fc4_buff(CFG::seqlen*CFG::ffdim);

    stage3_args_t s3_args;
    s3_args.fc_in = new int8_t[CFG::seqlen*CFG::dmodel];
    
    s3_args.dense_weight_t = stage3_dense_weight_t.data();
    s3_args.dense_bias = stage3_dense_bias.data();
    s3_args.dense_acc_scale = 0.004;
    s3_args.M_stage3 = 0.3;

    genmat(s3_args.fc_in, CFG::seqlen, CFG::dmodel, 5);

    for (int i = 0; i < CFG::dmodel; ++i) {
        for (int j = 0; j < CFG::seqlen; ++j) {
            stage3_fc_in.data()[i*CFG::seqlen+j] = stage3_args.fc_in[j*CFG::dmodel+i];
        }
    }

    genmat(s3_args.dense_weight_t, CFG::dmodel, CFG::ffdim, 13);
    genmat(s3_args. dense_bias, 1, CFG::ffdim, 71);

    /********** STAGE 4 ARGS ***********/
	std::vector<int8_t, aligned_allocator<int8_t>> stage4_dense_weight_t(CFG::ffdim*CFG::dmodel);
	std::vector<int8_t, aligned_allocator<int8_t>> stage4_dense_out(CFG::seqlen*CFG::dmodel);
	std::vector<int32_t, aligned_allocator<int32_t>> stage4_dense_bias(CFG::dmodel);
	std::vector<int16_t, aligned_allocator<int16_t>> stage4_norm_weight(CFG::dmodel);
	std::vector<int16_t, aligned_allocator<int16_t>> stage4_norm_bias(CFG::dmodel);
    
    auto stage4_out_gt = new int8_t[CFG::seqlen*CFG::dmodel];

    stage4_args_t s4_args;
    s4_args.skip_conn = s3_args.fc_in;
    s4_args.dense_weight_t = stage4_dense_weight_t.data();
    s4_args.norm_bias = stage4_norm_bias.data();
    s4_args.norm_weight = stage4_norm_weight.data();
    s4_args.dense_bias = stage4_dense_bias.data();
    s4_args.dense_out = stage4_dense_out.data();
    s4_args.M_residual = 2;
    s4_args.M_dense_acc = 1;
    s4_args.M_stage4 = 1;

    genmat(s4_args.dense_weight_t, CFG::ffdim, CFG::dmodel, 9);
    genmat(s4_args.dense_bias, CFG::dmodel, 1, 44);
    genmat(s4_args.norm_weight, CFG::dmodel, 1, 23);
    genmat(s4_args.norm_bias, CFG::dmodel, 1, 11);

    // Generate SW ground truth
    fpga2_gt(s3_args, s4_args);
    memcpy(stage4_out_gt, s4_args.dense_out, sizeof(int8_t)*CFG::seqlen*CFG::dmodel);
   
    // scramble output as sanity check that fpga is actually chaning memory 
    srand(time(NULL));
    for (int i = 0; i < CFG::seqlen*CFG::dmodel; ++i)
    {
        s4_args.dense_out[i] = rand() % 32767;
    }

    //printmat(stage4_out_gt, CFG::seqlen, CFG::dmodel);


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
      OCL_CHECK(err, krnl = cl::Kernel(program, "fpga2", &err));
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
  OCL_CHECK(err, cl::Buffer buffer_stage3_fc_in(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage3_fc_in)::value_type)*stage3_fc_in.size(),
                     stage3_fc_in.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage3_dense_weight_t(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage3_dense_weight_t)::value_type)*stage3_dense_weight_t.size(),
                     stage3_dense_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage3_dense_bias(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage3_dense_bias)::value_type)*stage3_dense_bias.size(),
                     stage3_dense_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_fc3_to_fc4_buff(
                     context, CL_MEM_USE_HOST_PTR, 
                     sizeof(decltype(fc3_to_fc4_buff)::value_type)*fc3_to_fc4_buff.size(),
                     fc3_to_fc4_buff.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_dense_weight_t(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage4_dense_weight_t)::value_type)*stage4_dense_weight_t.size(),
                     stage4_dense_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_dense_out(
                     context, CL_MEM_USE_HOST_PTR,
                     sizeof(decltype(stage4_dense_out)::value_type)*stage4_dense_out.size(),
                     stage4_dense_out.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_dense_bias(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage4_dense_bias)::value_type)*stage4_dense_bias.size(),
                     stage4_dense_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_norm_weight(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage4_norm_weight)::value_type)*stage4_norm_weight.size(),
                     stage4_norm_weight.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_norm_bias(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage4_norm_bias)::value_type)*stage4_norm_bias.size(),
                     stage4_norm_bias.data(), &err));


  OCL_CHECK(err, err = krnl.setArg(0, buffer_stage3_fc_in));
  OCL_CHECK(err, err = krnl.setArg(1, buffer_stage3_dense_weight_t));
  OCL_CHECK(err, err = krnl.setArg(2, buffer_stage3_dense_bias));
  OCL_CHECK(err, err = krnl.setArg(3, s3_args.dense_acc_scale));
  OCL_CHECK(err, err = krnl.setArg(4, s3_args.M_stage3));
  OCL_CHECK(err, err = krnl.setArg(5, buffer_fc3_to_fc4_buff));
  OCL_CHECK(err, err = krnl.setArg(6, buffer_stage4_dense_weight_t));
  OCL_CHECK(err, err = krnl.setArg(7, buffer_stage4_dense_out));
  OCL_CHECK(err, err = krnl.setArg(8, buffer_stage4_dense_bias));
  OCL_CHECK(err, err = krnl.setArg(9, buffer_stage4_norm_weight));
  OCL_CHECK(err, err = krnl.setArg(10, buffer_stage4_norm_bias));
  OCL_CHECK(err, err = krnl.setArg(11, s4_args.M_residual));
  OCL_CHECK(err, err = krnl.setArg(12, s4_args.M_dense_acc));
  OCL_CHECK(err, err = krnl.setArg(13, s4_args.M_stage4));

  // Copy input data to device global memory
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_stage3_fc_in, buffer_stage3_dense_weight_t, buffer_stage3_dense_bias,
                                                   buffer_stage4_dense_weight_t, buffer_stage4_dense_bias, buffer_stage4_norm_weight,
                                                   buffer_stage4_norm_bias},
                                                  0 ));

  q.finish();

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended
  // to always use enqueueTask() for invoking HLS kernel

  const int n_trials = 100;
  double cumu_time = 0;
  auto trial_times = std::vector<double>();
  for (int i = 0; i < n_trials; i++) {
	  double kernel_time_in_sec = 0;
	  std::chrono::duration<double> kernel_time(0);
	  auto kernel_start = std::chrono::high_resolution_clock::now();

	  // Only want to time this kernel to determine bandwidth
	  OCL_CHECK(err, err = q.enqueueTask(krnl));
	  q.finish();

	  auto kernel_end = std::chrono::high_resolution_clock::now();
	  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
	  kernel_time_in_sec = kernel_time.count();
	  cumu_time += kernel_time_in_sec;
	  trial_times.push_back(kernel_time_in_sec);

  // Copy Result from Device Global Memory to Host Local Memory
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_stage4_dense_out},
                                                  CL_MIGRATE_MEM_OBJECT_HOST));
  q.finish();

  }

  auto avg_time = cumu_time / n_trials;
  auto min_time = *std::min_element(trial_times.begin(), trial_times.end()); 
  std::cout << "MIN Exection time after " << n_trials << " trials = " << min_time*1000 << "ms" << std::endl;
  std::cout << "AVG Exection time after " << n_trials << " trials = " << avg_time*1000 << "ms" << std::endl;


  // OPENCL HOST CODE AREA END
  /*
  std::cout << "Ground truth" << std::endl;
  printmat(stage4_out_gt, CFG::seqlen, CFG::dmodel, "dense out GT");

  std::cout << "Test"<<std::endl;
  printmat(stage4_dense_out.data(), CFG::seqlen, CFG::dmodel, "dense out test");
  */

  // Compare the results of the Device to the simulation
  bool match = check(stage4_out_gt, stage4_dense_out.data(), CFG::seqlen, CFG::dmodel);

  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

