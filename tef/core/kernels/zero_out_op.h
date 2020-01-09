
#ifndef ZERO_OUT_OP_H_
#define ZERO_OUT_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
public:
  explicit ZeroOutOp(OpKernelConstruction * context);

public:
  void Compute(OpKernelContext* context) override;
};



#endif //ZERO_OUT_OP_H_

