#pragma once

#include <memory>
#include <string>

#include "zk_client.h"

/**
 * @brief 基于 ZooKeeper 的服务注册器。
 *
 * 该类负责按照约定的目录结构将 RPC 服务实例注册到 ZooKeeper，供服务发现
 * 或后续治理组件读取。
 */
class ServiceRegistry {
public:
    /**
     * @brief 使用共享的 ZooKeeper 客户端构造注册器。
     *
     * @param zk_client 已初始化的 ZooKeeper 客户端。
     */
    explicit ServiceRegistry(std::shared_ptr<ZkClient> zk_client);

    /**
     * @brief 注册一个服务实例节点。
     *
     * @param service_name 服务名，例如 `inference.InferenceService`。
     * @param endpoint 服务实例地址，例如 `127.0.0.1:50051`。
     * @param metadata 写入 ZooKeeper 节点的元信息。
     * @return 注册成功返回 true，否则返回 false。
     */
    bool Register(const std::string& service_name,
                  const std::string& endpoint,
                  const std::string& metadata);

private:
    std::shared_ptr<ZkClient> zk_client_;
};
