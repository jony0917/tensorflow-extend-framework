

#include "ps_client_factory.h"
#include "ps_client_dummy.h"


PsClient * PsClientFactory::Build(){
  return PsClientDummy::GetInstance();
}
