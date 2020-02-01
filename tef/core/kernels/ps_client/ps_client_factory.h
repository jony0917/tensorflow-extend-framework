
#ifndef PS_CLIENT_FACTORY_H
#define PS_CLIENT_FACTORY_H

#include "ps_client"

class PsClientFactory {
public:
  PsClient * Build();
};

#endif //PS_CLIENT_FACTORY_H