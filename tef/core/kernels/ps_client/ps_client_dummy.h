

#ifndef PS_CLIENT_DUMMY_H
#define PS_CLIENT_DUMMY_H

#include <mutex>
#include "ps_client.h"


class PsClientDummy : PsClient {
public:
  static PsClientDummy * GetInstance();

public:
  virtual void RegisterVariable(const VariableInfo& info, int &id) override;

  virtual void GetVariableInfo(const string& var_name, VariableInfo * var_info) override;

  virtual void DensePull(int id, Tensor* data) override;

  virtual void DensePush(int id,
                         const Tensor &data,
                         const std::string& updater) override;

  virtual void SparsePull(int id,
                            const Tensor &index,
                            Tensor* data) override;

  virtual void SparsePush(int id,
                          const Tensor& index,
                          const Tensor& data
                          const std::string& updater) override;

  virtual void HashPull(int id,
                        const Tensor& hash,
                        Tensor* data) override;


  virtual void HashPush(int id,
                        const Tensor& hash,
                        const Tensor& data,
                        const std::string& updater) override;

private:
  static std::mutex s_instance_mutex_;
};

#endif //PS_CLIENT_DUMMY_H