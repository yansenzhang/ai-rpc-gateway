#pragma once

#include <condition_variable>
#include <mutex>
#include <string>

#include <zookeeper/zookeeper.h>

class ZkClient {
public:
    ZkClient();
    ~ZkClient();

    ZkClient(const ZkClient&) = delete;
    ZkClient& operator=(const ZkClient&) = delete;

    // hosts：ZooKeeper 集群地址列表，格式如 host1:2181,host2:2181。
    // timeout_ms：连接超时时间，单位为毫秒。
    bool Connect(const std::string& hosts, int timeout_ms);
    void Close();

    bool IsConnected() const;

    // path：需要确保存在的持久节点路径。
    bool EnsurePath(const std::string& path);

    // path：要创建的临时节点完整路径。
    // value：节点数据内容。
    bool CreateEphemeral(const std::string& path, const std::string& value);

private:
    static void GlobalWatcher(zhandle_t* zh, int type, int state, const char* path, void* watcher_ctx);
    void HandleWatcher(int type, int state, const char* path);
    bool CreatePersistentNode(const std::string& path);

    zhandle_t* handle_;
    mutable std::mutex mutex_;
    std::condition_variable connected_cv_;
    bool connected_;
};
