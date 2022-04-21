#include "xcl2.hpp"
#include "cmdlineparser.h"
#include <iomanip>
#include <string>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include "pipeline.hpp"
#include "config.hpp"

// Declaration of custom p2p APIs that binds to Xilinx p2p APIs.
decltype(&xclGetMemObjectFd) xcl::P2P::getMemObjectFd = nullptr;
decltype(&xclGetMemObjectFromFd) xcl::P2P::getMemObjectFromFd = nullptr;

cl_program xcl_import_binary_file(cl_device_id device_id, cl_context context, const char* xclbin_file_name);


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

    /********* fpga1 SW Ground Truth *********/
    fpga1_gt(s1_args, s2_args);
    
    /********** STAGE 3 ARGS ***********/
	std::vector<int8_t, aligned_allocator<int8_t>> stage3_fc_in(CFG::seqlen*CFG::dmodel);
	std::vector<int8_t, aligned_allocator<int8_t>> stage3_dense_weight_t(CFG::dmodel*CFG::ffdim);
	std::vector<int32_t, aligned_allocator<int32_t>> stage3_dense_bias(CFG::ffdim);
	std::vector<int8_t, aligned_allocator<int8_t>> fc3_to_fc4_buff(CFG::seqlen*CFG::ffdim);

    stage3_args_t s3_args;
    s3_args.fc_in = s2_args.out; // For the SW ground truth output, chain fpga1 and fpga2 
    
    s3_args.dense_weight_t = stage3_dense_weight_t.data();
    s3_args.dense_bias = stage3_dense_bias.data();
    s3_args.dense_acc_scale = 0.004;
    s3_args.M_stage3 = 0.3;

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

    /********** fpga2 SW ground truth **********/
    fpga2_gt(s3_args, s4_args);
    memcpy(stage4_out_gt, s4_args.dense_out, sizeof(int8_t)*CFG::seqlen*CFG::dmodel);

    std::fill(stage2_out.begin(), stage2_out.end(), 0);
    std::fill(stage4_dense_out.begin(), stage4_dense_out.end(), 0);





    /**********************************/
    /* Get Devices and Create Kernels */
    
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file_krnl_fpga1", "-x1", "krnl_fpga1 binary file string", "");
    parser.addSwitch("--xclbin_file_krnl_fpga2", "-x2", "krnl_fpga2 binary file string", "");
    parser.parse(argc, argv);

    // Read settings
    auto binaryFile1 = parser.value("xclbin_file_krnl_fpga1");
    auto binaryFile2 = parser.value("xclbin_file_krnl_fpga2");

    if (argc != 5) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    cl_platform_id platform_id;
    cl_platform_id platforms[16] = {0};
    cl_uint platform_count;
    char platformName[256];
    cl_int error;

    clGetPlatformIDs(16, platforms, &platform_count);

    for (cl_uint i = 0; i < platform_count; i++) {
        error = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 256, platformName, 0);
        if (error != CL_SUCCESS) {
            exit(EXIT_FAILURE);
        }

        if (strcmp(platformName, "Xilinx") == 0) {
            platform_id = platforms[i];
        }
    }

    cl_uint device_count;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 0, nullptr, &device_count);
    std::cout << "Device count - " << device_count << std::endl;

    if (device_count < 2) {
        std::cout << "WARNING: This design does P2P transfer between two devices. "
                     "Please run this "
                     "design on machine with two devices.\n";
        return 0;
    }

    cl_device_id* device_id = (cl_device_id*)malloc(sizeof(cl_device_id) * device_count);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, device_count, device_id, nullptr);

    cl_bool is_nodma;
    uint8_t nodma_cnt = 0;
    clGetDeviceInfo(device_id[0], CL_DEVICE_NODMA, sizeof(is_nodma), &is_nodma, nullptr);
    if (is_nodma) nodma_cnt++;
    clGetDeviceInfo(device_id[1], CL_DEVICE_NODMA, sizeof(is_nodma), &is_nodma, nullptr);
    if (is_nodma) nodma_cnt++;

    if (xcl::is_hw_emulation()) {
        char device_name[256];
        clGetDeviceInfo(device_id[0], CL_DEVICE_NAME, 256, &device_name, nullptr);
        std::cout << device_name << std::endl;
        if (strstr(device_name, "2018") != 0) {
            std::cout << "[INFO]: The example is not supported for " << device_name
                      << " this platform for hw_emu. Please try other flows." << '\n';
            return EXIT_SUCCESS;
        }
        clGetDeviceInfo(device_id[1], CL_DEVICE_NAME, 256, &device_name, nullptr);
        if (strstr(device_name, "2018") != 0) {
            std::cout << "[INFO]: The example is not supported for " << device_name
                      << " this platform for hw_emu. Please try other flows." << '\n';
            return EXIT_SUCCESS;
        }
    }
    if (nodma_cnt == 2) {
        std::cout
            << "WARNING: P2P transfer can only be done between xdma and nodma devices but not between 2 nodma devices. "
               "Please run this "
               "design on machine with both xdma and nodma devices.\n";
        return 0;
    }
    

    cl::Context context[device_count];
    cl::CommandQueue queue[device_count];
    cl::Kernel krnl_fpga1, krnl_fpga2;
    cl::Program program[device_count];
    int err;

    std::chrono::high_resolution_clock::time_point p2pStart;
    std::chrono::high_resolution_clock::time_point p2pEnd;

    std::cout << "Initializing OpenCL objects" << std::endl;
    for (uint8_t i = 0; i < device_count; i++) {
        context[i] = clCreateContext(0, 1, &device_id[i], nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
            std::cout << "clCreateContext call: Failed to create a compute context" << err << std::endl;
        queue[i] = cl::CommandQueue(context[i], cl::Device(device_id[i]), CL_QUEUE_PROFILING_ENABLE, &err);
        if (err != CL_SUCCESS)
            std::cout << "clCreateCommandQueue call: Failed to create commandqueue" << err << std::endl;
    }

    //------------------------------- Program
    //-------------------------------------------
    auto fileBuf1 = xcl::read_binary_file(binaryFile1);
    cl::Program::Binaries bins1{{fileBuf1.data(), fileBuf1.size()}};
    program[0] = cl::Program(context[0], {cl::Device(device_id[0])}, bins1, NULL, &err);
    OCL_CHECK(err, krnl_fpga1= cl::Kernel(program[0], "fpga1", &err));
    auto fileBuf2 = xcl::read_binary_file(binaryFile2);
    cl::Program::Binaries bins2{{fileBuf2.data(), fileBuf2.size()}};
    program[1] = cl::Program(context[1], {cl::Device(device_id[1])}, bins2, NULL, &err);
    OCL_CHECK(err, krnl_fpga2= cl::Kernel(program[1], "fpga2", &err));

    //xcl::P2P::init(platform_id);
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();


  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  OCL_CHECK(err, cl::Buffer buffer_stage1_in(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_in)::value_type)*stage1_in.size(),
                     stage1_in.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_query(
                     context[0], CL_MEM_USE_HOST_PTR,
                     sizeof(decltype(query)::value_type)*query.size(),
                     query.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_key(
                     context[0], CL_MEM_USE_HOST_PTR,
                     sizeof(decltype(key)::value_type)*key.size(),
                     key.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_value(
                     context[0], CL_MEM_USE_HOST_PTR,
                     sizeof(decltype(value)::value_type)*value.size(),
                     value.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_query_weight_t(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_query_weight_t)::value_type)*stage1_query_weight_t.size(),
                     stage1_query_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_query_bias(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_query_bias)::value_type)*stage1_query_bias.size(),
                     stage1_query_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_key_weight_t(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_key_weight_t)::value_type)*stage1_key_weight_t.size(),
                     stage1_key_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_key_bias(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_key_bias)::value_type)*stage1_key_bias.size(),
                     stage1_key_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_value_weight_t(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_value_weight_t)::value_type)*stage1_value_weight_t.size(),
                     stage1_value_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage1_value_bias(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage1_value_bias)::value_type)*stage1_value_bias.size(),
                     stage1_value_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_out(
                     context[0], CL_MEM_USE_HOST_PTR,
                     sizeof(decltype(stage2_out)::value_type)*stage2_out.size(),
                     stage2_out.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_dense_weight_t(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage2_dense_weight_t)::value_type)*stage2_dense_weight_t.size(),
                     stage2_dense_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_dense_bias(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage2_dense_bias)::value_type)*stage2_dense_bias.size(),
                     stage2_dense_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_norm_weight(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage2_norm_weight)::value_type)*stage2_norm_weight.size(),
                     stage2_norm_weight.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage2_norm_bias(
                     context[0], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage2_norm_bias)::value_type)*stage2_norm_bias.size(),
                     stage2_norm_bias.data(), &err));


  OCL_CHECK(err, cl::Buffer buffer_stage3_fc_in(
                     context[1], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage3_fc_in)::value_type)*stage3_fc_in.size(),
                     stage3_fc_in.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage3_dense_weight_t(
                     context[1], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage3_dense_weight_t)::value_type)*stage3_dense_weight_t.size(),
                     stage3_dense_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage3_dense_bias(
                     context[1], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage3_dense_bias)::value_type)*stage3_dense_bias.size(),
                     stage3_dense_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_fc3_to_fc4_buff(
                     context[1], CL_MEM_USE_HOST_PTR, 
                     sizeof(decltype(fc3_to_fc4_buff)::value_type)*fc3_to_fc4_buff.size(),
                     fc3_to_fc4_buff.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_dense_weight_t(
                     context[1], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage4_dense_weight_t)::value_type)*stage4_dense_weight_t.size(),
                     stage4_dense_weight_t.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_dense_out(
                     context[1], CL_MEM_USE_HOST_PTR,
                     sizeof(decltype(stage4_dense_out)::value_type)*stage4_dense_out.size(),
                     stage4_dense_out.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_dense_bias(
                     context[1], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage4_dense_bias)::value_type)*stage4_dense_bias.size(),
                     stage4_dense_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_norm_weight(
                     context[1], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage4_norm_weight)::value_type)*stage4_norm_weight.size(),
                     stage4_norm_weight.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_stage4_norm_bias(
                     context[1], CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(decltype(stage4_norm_bias)::value_type)*stage4_norm_bias.size(),
                     stage4_norm_bias.data(), &err));



  OCL_CHECK(err, err = krnl_fpga1.setArg(0, buffer_stage1_in));
  OCL_CHECK(err, err = krnl_fpga1.setArg(1, buffer_query));
  OCL_CHECK(err, err = krnl_fpga1.setArg(2, buffer_key));
  OCL_CHECK(err, err = krnl_fpga1.setArg(3, buffer_value));
  OCL_CHECK(err, err = krnl_fpga1.setArg(4, buffer_stage1_query_weight_t));
  OCL_CHECK(err, err = krnl_fpga1.setArg(5, buffer_stage1_query_bias));
  OCL_CHECK(err, err = krnl_fpga1.setArg(6, buffer_stage1_key_weight_t));
  OCL_CHECK(err, err = krnl_fpga1.setArg(7, buffer_stage1_key_bias));
  OCL_CHECK(err, err = krnl_fpga1.setArg(8, buffer_stage1_value_weight_t));
  OCL_CHECK(err, err = krnl_fpga1.setArg(9, buffer_stage1_value_bias));
  OCL_CHECK(err, err = krnl_fpga1.setArg(10, s1_args.M_query));
  OCL_CHECK(err, err = krnl_fpga1.setArg(11, s1_args.M_key));
  OCL_CHECK(err, err = krnl_fpga1.setArg(12, s1_args.M_value));
  OCL_CHECK(err, err = krnl_fpga1.setArg(13, buffer_stage2_out));
  OCL_CHECK(err, err = krnl_fpga1.setArg(14, buffer_stage2_dense_weight_t));
  OCL_CHECK(err, err = krnl_fpga1.setArg(15, buffer_stage2_dense_bias));
  OCL_CHECK(err, err = krnl_fpga1.setArg(16, s2_args.M_attention_probs));
  OCL_CHECK(err, err = krnl_fpga1.setArg(17, s2_args.M_attention_out));
  OCL_CHECK(err, err = krnl_fpga1.setArg(18, s2_args.M_dense_out));
  OCL_CHECK(err, err = krnl_fpga1.setArg(19, s2_args.M_residual));
  OCL_CHECK(err, err = krnl_fpga1.setArg(20, buffer_stage2_norm_weight));
  OCL_CHECK(err, err = krnl_fpga1.setArg(21, buffer_stage2_norm_bias));
  OCL_CHECK(err, err = krnl_fpga1.setArg(22, s2_args.M_stage2));

  OCL_CHECK(err, err = krnl_fpga2.setArg(0, buffer_stage3_fc_in));
  OCL_CHECK(err, err = krnl_fpga2.setArg(1, buffer_stage3_dense_weight_t));
  OCL_CHECK(err, err = krnl_fpga2.setArg(2, buffer_stage3_dense_bias));
  OCL_CHECK(err, err = krnl_fpga2.setArg(3, s3_args.dense_acc_scale));
  OCL_CHECK(err, err = krnl_fpga2.setArg(4, s3_args.M_stage3));
  OCL_CHECK(err, err = krnl_fpga2.setArg(5, buffer_fc3_to_fc4_buff));
  OCL_CHECK(err, err = krnl_fpga2.setArg(6, buffer_stage4_dense_weight_t));
  OCL_CHECK(err, err = krnl_fpga2.setArg(7, buffer_stage4_dense_out));
  OCL_CHECK(err, err = krnl_fpga2.setArg(8, buffer_stage4_dense_bias));
  OCL_CHECK(err, err = krnl_fpga2.setArg(9, buffer_stage4_norm_weight));
  OCL_CHECK(err, err = krnl_fpga2.setArg(10, buffer_stage4_norm_bias));
  OCL_CHECK(err, err = krnl_fpga2.setArg(11, s4_args.M_residual));
  OCL_CHECK(err, err = krnl_fpga2.setArg(12, s4_args.M_dense_acc));
  OCL_CHECK(err, err = krnl_fpga2.setArg(13, s4_args.M_stage4));



  // Copy input data to device global memory
  OCL_CHECK(err, err = queue[0].enqueueMigrateMemObjects({buffer_stage1_in, buffer_query, buffer_key, buffer_value, buffer_stage1_query_weight_t, 
                                                   buffer_stage1_query_bias, buffer_stage1_key_weight_t, buffer_stage1_key_bias, 
                                                   buffer_stage1_value_weight_t, buffer_stage1_value_bias, buffer_stage2_dense_weight_t,
                                                   buffer_stage2_dense_bias, buffer_stage2_norm_weight, buffer_stage2_norm_bias},
                                                  0 ));

  OCL_CHECK(err, err = queue[1].enqueueMigrateMemObjects({buffer_stage3_dense_weight_t, buffer_stage3_dense_bias,
                                                   buffer_stage4_dense_weight_t, buffer_stage4_dense_bias, buffer_stage4_norm_weight,
                                                   buffer_stage4_norm_bias},
                                                  0 ));


    std::cout << "Launch FPGA-1\n" << std::endl;
    OCL_CHECK(err, err = queue[0].enqueueTask(krnl_fpga1));



    /*********** WITHOUT P2P **************/
    OCL_CHECK(err, err = queue[0].enqueueMigrateMemObjects({buffer_stage2_out},
                                                  CL_MIGRATE_MEM_OBJECT_HOST));

    queue[0].finish();

    memcpy(stage3_fc_in.data(), stage2_out.data(), sizeof(decltype(stage2_out)::value_type)*stage2_out.size()); 
    OCL_CHECK(err, err = queue[1].enqueueMigrateMemObjects({buffer_stage3_fc_in},
                                                  0 ));


/*
    //------------------------- P2P
    //-----------------------------------------------------------
    p2pStart = std::chrono::high_resolution_clock::now();
    std::cout << "Transferring from FPGA-1 to FPGA-2..." << std::endl;
    int fd = -1;
    OCL_CHECK(err, err = xcl::P2P::getMemObjectFd(buffer_stage3_fc_in(), &fd)); // Import p2p buffer to file descriptor (fd)
    if (fd > 0) {
        std::cout << "Import FD:" << fd << std::endl;
    }

    cl_mem exported_buf;
    OCL_CHECK(err, err = xcl::P2P::getMemObjectFromFd(context[0](), device_id[0], 0, fd, &exported_buf)); // Import
    cl_event event;
    OCL_CHECK(err,
              err = clEnqueueCopyBuffer(queue[0](), buffer_stage2_out(), exported_buf, 0, 0,sizeof(decltype(stage2_out)::value_type)*stage2_out.size(), 0, nullptr,
                                        &event)); // transfer
    clWaitForEvents(1, &event);
    p2pEnd = std::chrono::high_resolution_clock::now();
    clReleaseMemObject(exported_buf);
    // -----------------------------------------------------------------------
    */
    std::cout << "Launch FPGA-2\n" << std::endl;
    OCL_CHECK(err, err = queue[1].enqueueTask(krnl_fpga2));
    queue[1].finish();
    std::cout << "Read data back from FPGA-2 \n" << std::endl;
    OCL_CHECK(err, err = queue[1].enqueueMigrateMemObjects({buffer_stage4_dense_out},
                                                  CL_MIGRATE_MEM_OBJECT_HOST));
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    queue[0].finish();

    queue[1].finish();



    




/*
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

*/
  // OPENCL HOST CODE AREA END
  bool match_fpga1 = check(stage2_out_gt, stage2_out.data(), CFG::seqlen, CFG::dmodel);
 
  std::cout << "TEST FPGA1" << (match_fpga1 ? "PASSED" : "FAILED") << std::endl;

  std::cout << "Ground truth" << std::endl;
  printmat(stage4_out_gt, 10,1);

  std::cout << "Test"<<std::endl;
  printmat(stage4_dense_out.data(), 10,1);
  

  // Compare the results of the Device to the simulation
  bool match = check(stage4_out_gt, stage4_dense_out.data(), CFG::seqlen, CFG::dmodel);

  std::cout << "TEST FPGA2" << (match ? "PASSED" : "FAILED") << std::endl;

  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

