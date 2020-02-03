
#include "ps_sparse_pull_op.h"
#include "ps_client/ps_client_factory.h"

class PsSparsePullOp : public OpKernel {
public:
 explicit PsSparsePullOp(OpKernelConstruction* context) : OpKernel(context){
   OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
   OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
   OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
   CHECK(shape_.dims() >= 2);

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
   const Tensor &index = context->input(0);
   CHECK(index.dims() == 1)<<"index.dims="<< index.dims();

   Tensor* output_tensor = nullptr;
   TensorShape output_tensor_shape(index.shape());
   for(int i = 1; i < shape_.dims(); i++){
     output_tensor_shape.AddDim(shape_.dim_size(i));
   }
   OP_REQUIRES_OK(context, context->allocate_output(0, output_tensor_shape, &output_tensor));
   ps_client_->SparsePull(var_id_, index, output_tensor);
 }

private:
  TensorShape shape_;
  DataType dtype_;
  string var_name_;

  int var_id_;
  PsClient * ps_client_;
};


#define REGISTER_CPU_KERNEL(T) \
   REGISTER_KERNEL_BUILDER(Name("PsSparsePull").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), PsSparsePullOp);
REGISTER_CPU_KERNEL(bool)
REGISTER_CPU_KERNEL(int)
REGISTER_CPU_KERNEL(int64)
REGISTER_CPU_KERNEL(float)
REGISTER_CPU_KERNEL(double)

