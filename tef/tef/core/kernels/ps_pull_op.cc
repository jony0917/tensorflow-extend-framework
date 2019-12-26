
#include "ps_pull_op.h"



ZeroOutOp::ZeroOutOp(OpKernelConstruction* context) : OpKernel(context){
}


void ZeroOutOp::Compute(OpKernelContext* context) {
  // Grab the input tensor
  const Tensor& input_tensor = context->input(0);
  auto input = input_tensor.flat<int32>();

  // Create an output tensor
  Tensor* output_tensor = NULL;
  OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                   &output_tensor));
  auto output_flat = output_tensor->flat<int32>();

  // Set all but the first element of the output tensor to 0.
  const int N = input.size();
  for (int i = 1; i < N; i++) {
    output_flat(i) = 0;
  }

  // Preserve the first input value if possible.
  if (N > 0) output_flat(0) = input(0);
}

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);

