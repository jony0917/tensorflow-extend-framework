#include "ps_hash_pull_op.h"
#include "ps_client/ps_client_factory.h"

class PsHashPullOp : public OpKernel {
public:
 explicit PsHashPullOp(OpKernelConstruction* context) : OpKernel(context){
   OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
   OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
   OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));

   ps_client_ = PsClientFactory::Build();
   PsClient::VariableInfo var_info;
   var_info.var_name_ = var_name_;
   var_info.shape_ = shape_;
   var_info.dtype_ = dtype_;
   var_info.var_type_ = PsClient::VT_HASH;
   ps_client_->RegisterVariable(var_info, var_id_);
 }


public:
 void Compute(OpKernelContext* context) override {
   const Tensor &index = context->input(0);
   CHECK(index.dims() == 1)<<"index.dims="<< index.dims();

   Tensor* output_tensor = nullptr;
   TensorShape output_tensor_shape(index.shape());
   output_tensor_shape.AppendShape(shape_);
   OP_REQUIRES_OK(context, context->allocate_output(0, output_tensor_shape, &output_tensor));
   ps_client_->HashPull(var_id_, index, output_tensor);
 }

private:
  TensorShape shape_;
  DataType dtype_;
  string var_name_;
  int var_id_;
  PsClient * ps_client_;
};


#define REGISTER_CPU_KERNEL(T) \
   REGISTER_KERNEL_BUILDER(Name("PsHashPull").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), PsHashPullOp);
REGISTER_CPU_KERNEL(bool)
REGISTER_CPU_KERNEL(int)
REGISTER_CPU_KERNEL(int64)
REGISTER_CPU_KERNEL(float)
REGISTER_CPU_KERNEL(double)

