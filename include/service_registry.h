#pragma once

#include <memory>
#include <string>

#include "zk_client.h"

class ServiceRegistry {
public:
    explicit ServiceRegistry(std::shared_ptr<ZkClient> zk_client);

    // service_name：服务名，如 inference.InferenceService。
    // endpoint：服务实例地址，如 127.0.0.1:50051。
    // metadata：写入 ZooKeeper 节点的元信息。
    bool Register(const std::string& service_name,
                  const std::string& endpoint,
                  const std::string& metadata);

private:
    std::shared_ptr<ZkClient> zk_client_;
};
