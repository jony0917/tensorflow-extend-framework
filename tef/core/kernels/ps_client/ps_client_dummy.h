

#ifndef PS_CLIENT_DUMMY_H
#define PS_CLIENT_DUMMY_H

#include <mutex>
#include <vector>
#include <unordered_map>

#include "ps_client.h"


class PsClientDummy : public PsClient {
public:
  static PsClientDummy * GetInstance();

public:
  virtual void RegisterVariable(const VariableInfo& info, int &id) override;

  virtual void DensePull(int id, Tensor* data) override;

  virtual void DensePush(int id,
                         const Tensor &data,
                         const std::string& updater,
                         float learning_rate) override;

  virtual void SparsePull(int id,
                            const Tensor &index,
                            Tensor* data) override;

  virtual void SparsePush(int id,
                          const Tensor& index,
                          const Tensor& data,
                          const std::string& updater,
                          float learning_rate) override;

  virtual void HashPull(int id,
                        const Tensor& hash,
                        Tensor* data) override;


  virtual void HashPush(int id,
                        const Tensor& hash,
                        const Tensor& data,
                        const std::string& updater,
                        float learning_rate) override;

private:
  std::mutex register_variable_mutex_;
  static std::mutex s_instance_mutex_;

  struct Variable{
    Tensor dense_value_;
    std::unordered_map<int64, Tensor> hash_value_;
  };

  std::vector<Variable> variables_;
  std::vector<VariableInfo> variable_infos_;
  std::unordered_map<std::string, int> variable_ids_;

};

#endif //PS_CLIENT_DUMMY_H