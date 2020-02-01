
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("PsPull")
    .Attr("var_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Output("output: dtype");
REGISTER_OP("PsPush")
    .Input("v: dtype")
    .Attr("var_name: string")
    .Attr("dtype: type")
    .Attr("updater: string");



REGISTER_OP("PsSparsePull")
    .Input("index: index_type")
    .Attr("var_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("index_type: {int32, int64}")
    .Output("output: dtype");
REGISTER_OP("PsSparsePush")
    .Input("index: index_type")
    .Input("v: dtype")
    .Attr("var_name: string")
    .Attr("dtype: type")
    .Attr("index_type: {int32, int64}");




REGISTER_OP("PsHashPull")
    .Input("hash: hash_type")
    .Attr("var_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("hash_type: {int32, int64}")
    .Output("output: dtype");
REGISTER_OP("PsHashPush")
    .Input("hash: hash_type")
    .Input("v: dtype")
    .Attr("var_name: string")
    .Attr("dtype: type")
    .Attr("hash_type: {int32, int64}")
    .Output("output: dtype");
