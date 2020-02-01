

#include "ps_hash_push_op.h"


class PsHashPushOp : public OpKernel {
public:
 explicit PsHashPushOp(OpKernelConstruction* context) : OpKernel(context){
   OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
   OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
   OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
   OP_REQUIRES_OK(context, context->GetAttr("hash_type", &hash_type_));
   OP_REQUIRES_OK(context, context->GetAttr("updater", &updater_));

   ps_client_ = PsClientFactory::Build();
   PsClient::VariableInfo var_info;
   var_info.var_name_ = var_name;
   var_info.shape_ = shape_;
   var_info.dtype_ = dtype_;
   if(hash_type_ == DT_INT32){
     var_info.var_type_ = PsClient::VT_HASH32;
   }else{
     var_info.var_type_ = PsClient::VT_HASH64;
   }
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
   REGISTER_KERNEL_BUILDER(Name("PsHashPush").Device(DEVICE_CPU).TypeConstraint<T>("dtype"), PsHashPushOp);
REGISTER_CPU_KERNEL(bool)
REGISTER_CPU_KERNEL(int)
REGISTER_CPU_KERNEL(int64)
REGISTER_CPU_KERNEL(float)
REGISTER_CPU_KERNEL(double)

