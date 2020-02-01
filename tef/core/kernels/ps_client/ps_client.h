
#ifndef PS_CLIENT_H
#define PS_CLIENT_H

class PsClient {
public:
  virtual ~PsClient(){}

  enum VariableType{
    VT_DENSE = 0,
    VT_HASH32,
    VT_HASH64
  }

  struct VariableInfo{
    TensorShape shape_;
    DataType dtype_;
    string var_name_;
    VariableType var_type_;
  }


public:
  virtual void RegisterVariable(const VariableInfo& info, int &id) = 0;

  virtual void GetVariableInfo(const string& var_name, VariableInfo * var_info) = 0;

  virtual void DensePull(int id, Tensor* data) = 0;

  virtual void DensePush(int id,
                         const Tensor &data,
                         const std::string& updater) = 0;

  virtual void SparsePull(int id,
                          const Tensor &index,
                          Tensor* data) = 0;

  virtual void SparsePush(int id,
                          const Tensor& index,
                          const Tensor& data
                          const std::string& updater) = 0;

  virtual void HashPull(int id,
                        const Tensor& hash,
                        Tensor* data) = 0;


  virtual void HashPush(int id,
                        const Tensor& hash,
                        const Tensor& data,
                        const std::string& updater) = 0;

}

#endif //PS_CLIENT_H