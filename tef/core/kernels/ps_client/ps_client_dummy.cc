

#include "ps_client_dummy.h"
#include "tensorflow/core/framework/tensor_util.h"


namespace{
template<typename T>
void SGDUpdate(float alpha, const Tensor& gradient, Tensor * target){
  CHECK(target);
  CHECK(gradient.NumElements() == target->NumElements());

  auto target_vec = target->flat<T>();
  auto gradient_vec = gradient.flat<float>();
  for(int i = 0; i < target->NumElements(); i++){
    target_vec(i) -= alpha * gradient_vec(i);
  }
}


template<typename T>
void SGDUpdateSparse(float alpha, const Tensor& index, const Tensor& gradient, Tensor * target){
  auto index_vec = index.vec<int64>();
  auto target_matrix = target->flat_inner_dims<T>();
  auto gradient_matrix = gradient.flat_inner_dims<float>();
  CHECK(target_matrix.dimension(1) == gradient_matrix.dimension(1));

  for(int i = 0; i < index.NumElements(); i++){
    CHECK(index_vec(i) < target->dim_size(0));
    for(int j = 0; j < target_matrix.dimension(1); j++){
      target_matrix(index_vec(i), j) -= alpha * gradient_matrix(i, j);
    }
  }
}

template<typename T>
void SGDUpdateHash(float alpha, const Tensor& hash, const Tensor& gradient, std::unordered_map<int64, Tensor> * target){
  auto hash_vec = hash.vec<int64>();
  auto gradient_matrix = gradient.flat_inner_dims<float>();
  for(int i = 0; i < hash.NumElements(); i++){
    int64 key = hash_vec(i);
    auto it = target->find(key);
    CHECK(it != target->end());

    Tensor slice = it->second;
    auto slice_flat = slice.flat<T>();
    for(int j = 0; j < slice.NumElements(); j++){
      slice_flat(j) -= alpha * gradient_matrix(i, j);
    }
  }

}

template<typename T>
void LookUp(const Tensor& index, const Tensor& param, Tensor * out){

  auto index_vec = index.vec<int64>();
  auto out_matrix = out->flat_inner_dims<T>();
  auto param_matrix = param.flat_inner_dims<T>();
  CHECK(out_matrix.dimension(1) == param_matrix.dimension(1));

  for(int i = 0; i < index.NumElements(); i++){
    CHECK(index_vec(i) < param.dim_size(0));

    for(int j = 0; j < out_matrix.dimension(1); j++){
      out_matrix(i, j) = param_matrix(index_vec(i), j);
    }
  }
}

template<typename T>
void HashLookUp(const Tensor& hash, DataType dtype, const TensorShape shape, std::unordered_map<int64, Tensor> * param, Tensor * out){
  auto hash_vec = hash.vec<int64>();
  auto out_matrix = out->flat_inner_dims<T>();
  for(int i = 0; i < hash.NumElements(); i++){
    int64 key = hash_vec(i);
    auto it = param->find(key);
    if(it == param->end()){
      Tensor missing(dtype, shape);
      (*param)[key] = missing;
      auto missing_flat = missing.flat<T>();
      for(int j = 0; j < missing.NumElements(); j++){
        out_matrix(i, j) = missing_flat(j);
      }
    }else{
      Tensor slice = it->second;
      auto slice_flat = slice.flat<T>();
      for(int j = 0; j < it->second.NumElements(); j++){
        out_matrix(i, j) = slice_flat(j);
      }
    }
  }
}


}


std::mutex PsClientDummy::s_instance_mutex_;

//static
PsClientDummy * PsClientDummy::GetInstance(){
  static PsClientDummy * instance = nullptr;
  if (!instance){
    s_instance_mutex_.lock();
    if(!instance){
      instance = new PsClientDummy();
    }
    s_instance_mutex_.unlock();
   }
   return instance;
}

void PsClientDummy::RegisterVariable(const VariableInfo& info, int &id) {
  register_variable_mutex_.lock();
  auto it = variable_ids_.find(info.var_name_);
  if(it != variable_ids_.end()){
    id = it->second;
    CHECK(id < variable_infos_.size());
    CHECK(variable_infos_[id].shape_ == info.shape_);
    CHECK(variable_infos_[id].dtype_ == info.dtype_);
    CHECK(variable_infos_[id].var_type_ == info.var_type_);

  }else{
    id = variables_.size();
    variable_ids_[info.var_name_] = id;
    Variable var;
    if(info.var_type_ == VT_DENSE){
      var.dense_value_ = Tensor(info.dtype_, info.shape_);
    }
    variables_.push_back(var);
    variable_infos_.push_back(info);
  }

  register_variable_mutex_.unlock();
}

void PsClientDummy::DensePull(int id, Tensor* data) {
  CHECK(id < variables_.size());
  CHECK(variable_infos_[id].var_type_ == VT_DENSE);
  CHECK(variables_[id].dense_value_.NumElements() == data->NumElements());

  tensor::DeepCopy(variables_[id].dense_value_, data);
}

void PsClientDummy::DensePush(int id,
               const Tensor &data,
               const std::string& updater) {
  CHECK(id < variables_.size());
  CHECK(variable_infos_[id].var_type_ == VT_DENSE);
  CHECK(variables_[id].dense_value_.NumElements() == data.NumElements());
  CHECK(updater == "SGD");
  switch (variable_infos_[id].dtype_){
    case DT_FLOAT:
      SGDUpdate<float>(0.001, data, &variables_[id].dense_value_);
      break;
    case DT_DOUBLE:
      SGDUpdate<double>(0.001, data, &variables_[id].dense_value_);
      break;
    case DT_INT32:
      SGDUpdate<int>(0.001, data, &variables_[id].dense_value_);
      break;
    case DT_INT64:
      SGDUpdate<int64>(0.001, data, &variables_[id].dense_value_);
      break;
    default:
      break;
  }
}

void PsClientDummy::SparsePull(int id,
                const Tensor &index,
                Tensor* data) {
  CHECK(id < variables_.size());
  CHECK(variable_infos_[id].var_type_ == VT_DENSE);
  switch (variable_infos_[id].dtype_){
    case DT_FLOAT:
      LookUp<float>(index, variables_[id].dense_value_, data);
      break;
    case DT_DOUBLE:
      LookUp<double>(index, variables_[id].dense_value_, data);
      break;
    case DT_INT32:
      LookUp<int>(index, variables_[id].dense_value_, data);
      break;
    case DT_INT64:
      LookUp<int64>(index, variables_[id].dense_value_, data);
      break;
    default:
      break;
  }


}

void PsClientDummy::SparsePush(int id,
                const Tensor& index,
                const Tensor& data,
                const std::string& updater) {
  CHECK(id < variables_.size());
  CHECK(variable_infos_[id].var_type_ == VT_DENSE);
  CHECK(updater == "SGD");
  switch (variable_infos_[id].dtype_){
    case DT_FLOAT:
      SGDUpdateSparse<float>(0.001, index, data, &variables_[id].dense_value_);
      break;
    case DT_DOUBLE:
      SGDUpdateSparse<double>(0.001, index, data, &variables_[id].dense_value_);
      break;
    case DT_INT32:
      SGDUpdateSparse<int>(0.001, index, data, &variables_[id].dense_value_);
      break;
    case DT_INT64:
      SGDUpdateSparse<int64>(0.001, index, data, &variables_[id].dense_value_);
      break;
    default:
      break;
  }
}


void PsClientDummy::HashPull(int id,
              const Tensor& hash,
              Tensor* data) {
  CHECK(id < variables_.size());
  CHECK(variable_infos_[id].var_type_ == VT_HASH);
  switch (variable_infos_[id].dtype_){
    case DT_FLOAT:
      HashLookUp<float>(hash, variable_infos_[id].dtype_, variable_infos_[id].shape_, &variables_[id].hash_value_, data);
      break;
    case DT_DOUBLE:
      HashLookUp<double>(hash, variable_infos_[id].dtype_, variable_infos_[id].shape_, &variables_[id].hash_value_, data);
      break;
    case DT_INT32:
      HashLookUp<int>(hash, variable_infos_[id].dtype_, variable_infos_[id].shape_, &variables_[id].hash_value_, data);
      break;
    case DT_INT64:
      HashLookUp<int64>(hash, variable_infos_[id].dtype_, variable_infos_[id].shape_, &variables_[id].hash_value_, data);
      break;
    default:
      break;
  }

}


void PsClientDummy::HashPush(int id,
              const Tensor& hash,
              const Tensor& data,
              const std::string& updater) {
  CHECK(id < variables_.size());
  CHECK(variable_infos_[id].var_type_ == VT_HASH);
  CHECK(updater == "SGD");
  switch (variable_infos_[id].dtype_){
    case DT_FLOAT:
      SGDUpdateHash<float>(0.001, hash, data, &variables_[id].hash_value_);
      break;
    case DT_DOUBLE:
      SGDUpdateHash<double>(0.001, hash, data, &variables_[id].hash_value_);
      break;
    case DT_INT32:
      SGDUpdateHash<int>(0.001, hash, data, &variables_[id].hash_value_);
      break;
    case DT_INT64:
      SGDUpdateHash<int64>(0.001, hash, data, &variables_[id].hash_value_);
      break;
    default:
      break;
  }
}

