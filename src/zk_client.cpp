#include "zk_client.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

namespace {

// 将完整 ZooKeeper 路径拆分为逐级创建节点所需的路径片段。
std::vector<std::string> SplitPath(const std::string& path) {
    std::vector<std::string> segments;
    std::stringstream stream(path);
    std::string segment;
    while (std::getline(stream, segment, '/')) {
        if (!segment.empty()) {
            segments.push_back(segment);
        }
    }
    return segments;
}

}  // namespace

ZkClient::ZkClient()
    : handle_(nullptr), connected_(false) {
}

ZkClient::~ZkClient() {
    Close();
}

bool ZkClient::Connect(const std::string& hosts, int timeout_ms) {
    Close();

    connected_ = false;
    handle_ = zookeeper_init(hosts.c_str(), &ZkClient::GlobalWatcher, timeout_ms, nullptr, this, 0);
    if (handle_ == nullptr) {
        std::cerr << "[ZkClient] Failed to initialize ZooKeeper client" << std::endl;
        return false;
    }

    // 通过条件变量等待 watcher 汇报连接状态，避免调用方立即使用尚未就绪的句柄。
    std::unique_lock<std::mutex> lock(mutex_);
    bool connected = connected_cv_.wait_for(lock,
                                            std::chrono::milliseconds(timeout_ms),
                                            [this] { return connected_; });
    if (!connected) {
        std::cerr << "[ZkClient] Timed out while waiting for ZooKeeper connection" << std::endl;
        return false;
    }

    std::cout << "[ZkClient] Connected to ZooKeeper" << std::endl;
    return true;
}

void ZkClient::Close() {
    if (handle_ != nullptr) {
        zookeeper_close(handle_);
        handle_ = nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    connected_ = false;
}

bool ZkClient::IsConnected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return connected_;
}

bool ZkClient::EnsurePath(const std::string& path) {
    auto segments = SplitPath(path);
    std::string current_path;
    for (const auto& segment : segments) {
        current_path += "/" + segment;
        if (!CreatePersistentNode(current_path)) {
            return false;
        }
    }
    return true;
}

bool ZkClient::CreateEphemeral(const std::string& path, const std::string& value) {
    if (handle_ == nullptr) {
        return false;
    }

    int rc = zoo_create(handle_,
                        path.c_str(),
                        value.data(),
                        static_cast<int>(value.size()),
                        &ZOO_OPEN_ACL_UNSAFE,
                        ZOO_EPHEMERAL,
                        nullptr,
                        0);
    if (rc == ZNODEEXISTS) {
        return true;
    }
    if (rc != ZOK) {
        std::cerr << "[ZkClient] Failed to create ephemeral node " << path << ", rc=" << rc << std::endl;
        return false;
    }
    return true;
}

void ZkClient::GlobalWatcher(zhandle_t* zh, int type, int state, const char* path, void* watcher_ctx) {
    (void)zh;
    auto* client = static_cast<ZkClient*>(watcher_ctx);
    if (client != nullptr) {
        client->HandleWatcher(type, state, path);
    }
}

void ZkClient::HandleWatcher(int type, int state, const char* path) {
    (void)type;
    (void)path;

    std::lock_guard<std::mutex> lock(mutex_);

    // 只要进入可读写或只读连接态，就视为底层会话已经建立。
    connected_ = (state == ZOO_CONNECTED_STATE || state == ZOO_READONLY_STATE);
    if (connected_) {
        connected_cv_.notify_all();
    }
}

bool ZkClient::CreatePersistentNode(const std::string& path) {
    if (handle_ == nullptr) {
        return false;
    }

    Stat stat;
    int exists_rc = zoo_exists(handle_, path.c_str(), 0, &stat);
    if (exists_rc == ZOK) {
        return true;
    }
    if (exists_rc != ZNONODE) {
        std::cerr << "[ZkClient] Failed to check path " << path << ", rc=" << exists_rc << std::endl;
        return false;
    }

    // 对不存在的层级节点按需创建持久节点，供后续实例临时节点挂载。
    int create_rc = zoo_create(handle_,
                               path.c_str(),
                               "",
                               0,
                               &ZOO_OPEN_ACL_UNSAFE,
                               ZOO_PERSISTENT,
                               nullptr,
                               0);
    if (create_rc == ZNODEEXISTS) {
        return true;
    }
    if (create_rc != ZOK) {
        std::cerr << "[ZkClient] Failed to create persistent node " << path << ", rc=" << create_rc << std::endl;
        return false;
    }
    return true;
}
