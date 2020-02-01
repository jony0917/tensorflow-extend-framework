

#include "ps_client_dummy.h"

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

}

void PsClientDummy::GetVariableInfo(const string& var_name, VariableInfo * var_info) {

}

void PsClientDummy::DensePull(int id, Tensor* data) {

}

void PsClientDummy::DensePush(int id,
               const Tensor &data,
               const std::string& updater) {

}

void PsClientDummy::SparsePull(int id,
                const Tensor &index,
                Tensor* data) {
}

void PsClientDummy::SparsePush(int id,
                const Tensor& index,
                const Tensor& data,
                const std::string& updater) {
}


void PsClientDummy::HashPull(int id,
              const Tensor& hash,
              Tensor* data) {

}


void PsClientDummy::HashPush(int id,
              const Tensor& hash,
              const Tensor& data,
              const std::string& updater) {

}

