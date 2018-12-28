// #ifndef USE_ARMCL
// #error "This example needs to be built with -DUSE_ARMCL"
// #endif /* USE_ARMCL */

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "utils/Utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace armcl = arm_compute;
namespace armcl_utils = arm_compute::utils;

class ARMCLTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
    // Code here will be called immediately after each test
    // (right before the destructor).
  }
};


TEST_F(ARMCLTest,matrix_mult)
{
  int n = 3;
  int m = 2;
  int p = 4;

  std::vector<float> src_a = {2, 1,
                              6, 4,
                              2, 3};
  std::vector<float> src_b = {5, 2, 1, 6,
                              3, 7, 4, 1};

  std::vector<float> c_targets = {13, 11, 6, 13,
                                  42, 40, 22, 40,
                                  19, 25, 14, 15};

  // Provides global access to a CL context and command queue.
  armcl::CLTuner tuner{};
  armcl::CLScheduler::get().default_init(&tuner);

  armcl::CLTensor a{}, b{}, c{};
  float alpha = 1;
  float beta = 0;
  // Initialize the tensors dimensions and type:
  const armcl::TensorShape shape_a(m, n, 1);
  const armcl::TensorShape shape_b(p, m, 1);
  const armcl::TensorShape shape_c(p, n, 1);
  a.allocator()->init(armcl::TensorInfo(shape_a, 1, armcl::DataType::F32));
  b.allocator()->init(armcl::TensorInfo(shape_b, 1, armcl::DataType::F32));
  c.allocator()->init(armcl::TensorInfo(shape_c, 1, armcl::DataType::F32));

  // configure sgemm
  armcl::CLGEMM sgemm{};
  sgemm.configure(&a, &b, nullptr, &c, alpha, beta);

  // // Allocate the input / output tensors:
  a.allocator()->allocate();
  b.allocator()->allocate();
  c.allocator()->allocate();

  armcl::Window input_window;
  a.map();
  armcl::Iterator input_it(&a, input_window);
  input_window.use_tensor_dimensions(shape_a);

  // // Fill the input tensor:
  // // Simplest way: create an iterator to iterate through each element of the input tensor:
  // for( unsigned int y = 0; y < m; y++)
  //     for( unsigned int x = 0; x < n; x++)
  //         *reinterpret_cast<float *>(a.buffer() + a.info()->offset_element_in_bytes(armcl::Coordinates(y, x, 0))) = src_a[x * m + y];
  // Except it works for an arbitrary number of dimensions
  execute_window_loop(input_window, [&](const armcl::Coordinates & id)
                      {
                        *reinterpret_cast<float *>(input_it.ptr()) = src_a[id.z() * (m * n) + id.y() * m + id.x()];
                      },
                      input_it);
  a.unmap();

  armcl::Window input_window_b;
  b.map();
  armcl::Iterator input_it_b(&b, input_window_b);
  input_window_b.use_tensor_dimensions(shape_b);

  execute_window_loop(input_window_b, [&](const armcl::Coordinates & id)
                      {
                        *reinterpret_cast<float *>(input_it_b.ptr()) = src_b[id.z() * (p * m) + id.y() * p + id.x()];
                      },
                      input_it_b);

  b.unmap();

  // Configure function
  armcl_utils::init_sgemm_output(c, a, b, armcl::DataType::F32);

  // Dummy run for CLTuner
  sgemm.run();

  // Make sure all the OpenCL jobs are done executing:
  armcl::CLScheduler::get().sync();

  c.map();
  std::vector<float> lin_c(n * p);
  armcl::Window output_window;
  armcl::Iterator output_it(&c, output_window);
  output_window.use_tensor_dimensions(shape_c);
  for( unsigned int y = 0; y < p; y++)
    for( unsigned int x = 0; x < n; x++)
      lin_c[x * p + y] = *reinterpret_cast<float *>(c.buffer() + c.info()->offset_element_in_bytes(armcl::Coordinates(y, x, 0)));

  c.unmap();

  EXPECT_EQ(c_targets, lin_c);
}
