#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include "pipeline.hpp"
#include "config.hpp"

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


    /********** STAGE 1 ARGS ***********/
    std::vector<int8_t, aligned_allocator<int8_t>> stage1_in(CFG::seqlen*CFG::dmodel);
    std::vector<int8_t, aligned_allocator<int8_t>> stage1_query_weight_t(CFG::dmodel*CFG::dmodel);
    std::vector<int8_t, aligned_allocator<int8_t>> stage1_key_weight_t(CFG::dmodel*CFG::dmodel);
    std::vector<int8_t, aligned_allocator<int8_t>> stage1_value_weight_t(CFG::dmodel*CFG::dmodel);
    std::vector<int32_t, aligned_allocator<int32_t>> stage1_query_bias(CFG::dmodel);
    std::vector<int32_t, aligned_allocator<int32_t>> stage1_key_bias(CFG::dmodel);
    std::vector<int32_t, aligned_allocator<int32_t>> stage1_value_bias(CFG::dmodel);

    std::vector<int8_t, aligned_allocator<int8_t>> query(CFG::seqlen*CFG::dmodel);
    std::vector<int8_t, aligned_allocator<int8_t>> key(CFG::seqlen*CFG::dmodel);
    std::vector<int8_t, aligned_allocator<int8_t>> value(CFG::seqlen*CFG::dmodel);

    stage1_args_t s1_args;
    s1_args.in = stage1_in.data();
    s1_args.query_weight_t = stage1_query_weight_t.data();
    s1_args.key_weight_t = stage1_key_weight_t.data();
    s1_args.value_weight_t = stage1_value_weight_t.data();
    s1_args.query_bias = stage1_query_bias.data();
    s1_args.key_bias = stage1_key_bias.data();
    s1_args.value_bias = stage1_value_bias.data();
    s1_args.M_query = 0.5;
    s1_args.M_key = 0.4;
    s1_args.M_value = 0.3;

    genmat(s1_args.in, CFG::seqlen, CFG::dmodel, 7);
    genmat(s1_args.query_weight_t, CFG::dmodel, CFG::dmodel, 9);
    genmat(s1_args.key_weight_t, CFG::dmodel, CFG::dmodel, 11);
    genmat(s1_args.value_weight_t, CFG::dmodel, CFG::dmodel, 13);
    genmat(s1_args.query_bias, 1, CFG::dmodel, 63);
    genmat(s1_args.key_bias, 1, CFG::dmodel, 65);
    genmat(s1_args.value_bias, 1, CFG::dmodel, 67);

    /********** STAGE 2 ARGS ***********/
    std::vector<int8_t, aligned_allocator<int8_t>> stage2_out(CFG::seqlen*CFG::dmodel);
    std::vector<int8_t, aligned_allocator<int8_t>> stage2_dense_weight_t(CFG::dmodel*CFG::dmodel);
    std::vector<int32_t, aligned_allocator<int32_t>> stage2_dense_bias(CFG::dmodel);
    std::vector<int16_t, aligned_allocator<int16_t>> stage2_norm_weight(CFG::dmodel);
    std::vector<int16_t, aligned_allocator<int16_t>> stage2_norm_bias(CFG::dmodel);

    auto stage2_out_gt = new int8_t[CFG::seqlen*CFG::dmodel];

    stage2_args_t s2_args;
    s2_args.out = stage2_out_gt;
    s2_args.dense_weight_t = stage2_dense_weight_t.data();
    s2_args.dense_bias = stage2_dense_bias.data();
    s2_args.norm_weight = stage2_norm_weight.data();
    s2_args.norm_bias = stage2_norm_bias.data();
    s2_args.M_attention_probs = 100;
    s2_args.M_attention_out = 0.1;
    s2_args.M_dense_out = 0.1;
    s2_args.M_residual = 1;
    s2_args.M_stage2 = 1;

    genmat(s2_args.dense_weight_t, CFG::dmodel, CFG::dmodel, 13);
    genmat(s2_args.dense_bias, CFG::dmodel, 1, 61);
    genmat(s2_args.norm_weight, CFG::dmodel, 1, 62);
    genmat(s2_args.norm_bias, CFG::dmodel, 1, 69);

    /********* SW Ground Truth *********/
    fpga1_gt(s1_args, s2_args);
    
    // Double double check 
    std::fill(stage2_out.begin(), stage2_out.end(), 0);

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
      OCL_CHECK(err, krnl = cl::Kernel(program, "fpga1", &err));
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
  OCL_CHECK(err, cl::Buffer buffer_stage1_in(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_in)::value_type)*stage1_in.size(),
                     stage1_in.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_query(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(query)::value_type)*query.size(),
                     query.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_key(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(key)::value_type)*key.size(),
                     key.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_value(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(value)::value_type)*value.size(),
                     value.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_query_weight_t(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_query_weight_t)::value_type)*stage1_query_weight_t.size(),
                     stage1_query_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_query_bias(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_query_bias)::value_type)*stage1_query_bias.size(),
                     stage1_query_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_key_weight_t(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_key_weight_t)::value_type)*stage1_key_weight_t.size(),
                     stage1_key_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_key_bias(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_key_bias)::value_type)*stage1_key_bias.size(),
                     stage1_key_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_value_weight_t(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_value_weight_t)::value_type)*stage1_value_weight_t.size(),
                     stage1_value_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_value_bias(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_value_bias)::value_type)*stage1_value_bias.size(),
                     stage1_value_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_out(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage2_out)::value_type)*stage2_out.size(),
                     stage2_out.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_dense_weight_t(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage2_dense_weight_t)::value_type)*stage2_dense_weight_t.size(),
                     stage2_dense_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_dense_bias(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage2_dense_bias)::value_type)*stage2_dense_bias.size(),
                     stage2_dense_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_norm_weight(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage2_norm_weight)::value_type)*stage2_norm_weight.size(),
                     stage2_norm_weight.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_norm_bias(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage2_norm_bias)::value_type)*stage2_norm_bias.size(),
                     stage2_norm_bias.data(), &err));


  OCL_CHECK(err, err = krnl.setArg(0, buffer_stage1_in));
  OCL_CHECK(err, err = krnl.setArg(1, buffer_query));
  OCL_CHECK(err, err = krnl.setArg(2, buffer_key));
  OCL_CHECK(err, err = krnl.setArg(3, buffer_value));
  OCL_CHECK(err, err = krnl.setArg(4, buffer_stage1_query_weight_t));
  OCL_CHECK(err, err = krnl.setArg(5, buffer_stage1_query_bias));
  OCL_CHECK(err, err = krnl.setArg(6, buffer_stage1_key_weight_t));
  OCL_CHECK(err, err = krnl.setArg(7, buffer_stage1_key_bias));
  OCL_CHECK(err, err = krnl.setArg(8, buffer_stage1_value_weight_t));
  OCL_CHECK(err, err = krnl.setArg(9, buffer_stage1_value_bias));
  OCL_CHECK(err, err = krnl.setArg(10, s1_args.M_query));
  OCL_CHECK(err, err = krnl.setArg(11, s1_args.M_key));
  OCL_CHECK(err, err = krnl.setArg(12, s1_args.M_value));
  OCL_CHECK(err, err = krnl.setArg(13, buffer_stage2_out));
  OCL_CHECK(err, err = krnl.setArg(14, buffer_stage2_dense_weight_t));
  OCL_CHECK(err, err = krnl.setArg(15, buffer_stage2_dense_bias));
  OCL_CHECK(err, err = krnl.setArg(16, s2_args.M_attention_probs));
  OCL_CHECK(err, err = krnl.setArg(17, s2_args.M_attention_out));
  OCL_CHECK(err, err = krnl.setArg(18, s2_args.M_dense_out));
  OCL_CHECK(err, err = krnl.setArg(19, s2_args.M_residual));
  OCL_CHECK(err, err = krnl.setArg(20, buffer_stage2_norm_weight));
  OCL_CHECK(err, err = krnl.setArg(21, buffer_stage2_norm_bias));
  OCL_CHECK(err, err = krnl.setArg(22, s2_args.M_stage2));

  // Copy input data to device global memory
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_stage1_in, buffer_query, buffer_key, buffer_value, buffer_stage1_query_weight_t, 
                                                   buffer_stage1_query_bias, buffer_stage1_key_weight_t, buffer_stage1_key_bias, 
                                                   buffer_stage1_value_weight_t, buffer_stage1_value_bias, buffer_stage2_dense_weight_t,
                                                   buffer_stage2_dense_bias, buffer_stage2_norm_weight, buffer_stage2_norm_bias},
                                                  0 ));

  q.finish();

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended
  // to always use enqueueTask() for invoking HLS kernel

  const int n_trials = 1;
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
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_stage2_out},
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
  bool match = check(stage2_out_gt, stage2_out.data(), CFG::seqlen, CFG::dmodel);

  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

