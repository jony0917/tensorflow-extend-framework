

#include "ps_sparse_push_op.h"

class PsSparsePushOp : public OpKernel {
public:
 explicit PsSparsePushOp(OpKernelConstruction* context) : OpKernel(context){
   OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
   OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
   OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
   OP_REQUIRES_OK(context, context->GetAttr("updater", &updater_));

   ps_client_ = PsClientFactory::Build();
   PsClient::VariableInfo var_info;
   var_info.var_name_ = var_name;
   var_info.shape_ = shape_;
   var_info.dtype_ = dtype_;
   var_info.var_type_ = PsClient::VT_DENSE;
   ps_client_->RegisterVariable(var_info, var_id_);
 }


public:
 void Compute(OpKernelContext* context) override {
   const Tensor &index = context->input(0);
   const Tensor &data = context->input(1);
   ps_client_->SparsePush(var_id_, index, data, updater_);
 }

private:
  TensorShape shape_;
  DataType dtype_;
  std::string var_name_;
  std::string updater_;

  int var_id_;
  PsClient * ps_client_;
};



#define REGISTER_CPU_KERNEL(T) \
   REGISTER_KERNEL_BUILDER(Name("PsSparsePush").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), PsSparsePushOp);
REGISTER_CPU_KERNEL(bool)
REGISTER_CPU_KERNEL(int)
REGISTER_CPU_KERNEL(int64)
REGISTER_CPU_KERNEL(float)
REGISTER_CPU_KERNEL(double)

