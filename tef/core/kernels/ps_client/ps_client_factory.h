
#ifndef PS_CLIENT_FACTORY_H
#define PS_CLIENT_FACTORY_H

#include "ps_client.h"

class PsClientFactory {
public:
  static PsClient * Build();
};

#endif //PS_CLIENT_FACTORY_H