#include "service_registry.h"

#include <iostream>

ServiceRegistry::ServiceRegistry(std::shared_ptr<ZkClient> zk_client)
    : zk_client_(std::move(zk_client)) {
}

bool ServiceRegistry::Register(const std::string& service_name,
                               const std::string& endpoint,
                               const std::string& metadata) {
    if (!zk_client_ || !zk_client_->IsConnected()) {
        std::cerr << "[ServiceRegistry] ZooKeeper is not connected, skip registration" << std::endl;
        return false;
    }

    // 统一采用 /ai-rpc-gateway/services/<service>/providers/<endpoint> 目录结构注册实例。
    const std::string root_path = "/ai-rpc-gateway/services";
    const std::string service_path = root_path + "/" + service_name;
    const std::string providers_path = service_path + "/providers";
    const std::string endpoint_path = providers_path + "/" + endpoint;

    if (!zk_client_->EnsurePath(root_path) ||
        !zk_client_->EnsurePath(service_path) ||
        !zk_client_->EnsurePath(providers_path)) {
        return false;
    }

    // 具体实例使用临时节点承载，便于连接断开后由 ZooKeeper 自动清理。
    if (!zk_client_->CreateEphemeral(endpoint_path, metadata)) {
        return false;
    }

    std::cout << "[ServiceRegistry] Registered service endpoint " << endpoint_path << std::endl;
    return true;
}
