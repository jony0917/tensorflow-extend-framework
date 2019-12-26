

#ifndef PS_PULL_OP_H
#define PS_PULL_OP_H

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public Opkernel {
public:
 explicit ZeroOutOp(OpKernelConstruction* context);

public:
 void Compute(OpKernelContext* context) override;
};

#endif //PS_PULL_OP_H

