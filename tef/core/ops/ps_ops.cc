
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("PsPull")
    .Attr("var_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Output("output: dtype");


REGISTER_OP("PsPush")
    .Input("g: float")
    .Attr("var_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("updater: string")
    .Attr("learning_rate: float");



REGISTER_OP("PsSparsePull")
    .Input("index: int64")
    .Attr("var_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Output("output: dtype");
REGISTER_OP("PsSparsePush")
    .Input("index: int64")
    .Input("g: float")
    .Attr("var_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("updater: string")
    .Attr("learning_rate: float");




REGISTER_OP("PsHashPull")
    .Input("hash: int64")
    .Attr("var_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Output("output: dtype");
REGISTER_OP("PsHashPush")
    .Input("hash: int64")
    .Input("g: float")
    .Attr("var_name: string")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("updater: string")
    .Attr("learning_rate: float");
