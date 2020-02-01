
#include "ps_push_op.h"
#include "ps_client/ps_client_factory.h"

class PsPushOp : public OpKernel {
public:
 explicit PsPushOp(OpKernelConstruction* context) : OpKernel(context){
   OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
   OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
   OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
   OP_REQUIRES_OK(context, context->GetAttr("updater", &updater_));

   ps_client_ = PsClientFactory::Build();
   PsClient::VariableInfo var_info;
   var_info.var_name_ = var_name_;
   var_info.shape_ = shape_;
   var_info.dtype_ = dtype_;
   var_info.var_type_ = PsClient::VT_DENSE;
   ps_client_->RegisterVariable(var_info, var_id_);
 }


public:
 void Compute(OpKernelContext* context) override {
   const Tensor &data = context->input(0);
   ps_client_->DensePush(var_id_, data, updater_);
 }


private:
  TensorShape shape_;
  DataType dtype_;
  string var_name_;
  string updater_;

  int var_id_;
  PsClient * ps_client_;
};




#define REGISTER_CPU_KERNEL(T) \
   REGISTER_KERNEL_BUILDER(Name("PsPush").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), PsPushOp);
REGISTER_CPU_KERNEL(bool)
REGISTER_CPU_KERNEL(int)
REGISTER_CPU_KERNEL(int64)
REGISTER_CPU_KERNEL(float)
REGISTER_CPU_KERNEL(double)

