#pragma once

#include <condition_variable>
#include <mutex>
#include <string>

#include <zookeeper/zookeeper.h>

/**
 * @brief 对 ZooKeeper C 客户端的轻量封装。
 *
 * 该类负责管理 ZooKeeper 连接生命周期、连接状态同步、持久路径创建以及
 * 临时节点注册。其主要用途是为服务注册模块提供最小可用的节点操作能力。
 */
class ZkClient {
public:
    /**
     * @brief 构造一个未连接的 ZooKeeper 客户端。
     */
    ZkClient();

    /**
     * @brief 析构客户端并关闭底层连接。
     */
    ~ZkClient();

    ZkClient(const ZkClient&) = delete;
    ZkClient& operator=(const ZkClient&) = delete;

    /**
     * @brief 连接到指定的 ZooKeeper 集群。
     *
     * @param hosts ZooKeeper 集群地址列表，格式如 `host1:2181,host2:2181`。
     * @param timeout_ms 连接超时时间，单位为毫秒。
     * @return 连接建立成功返回 true，否则返回 false。
     */
    bool Connect(const std::string& hosts, int timeout_ms);

    /**
     * @brief 主动关闭当前 ZooKeeper 连接。
     */
    void Close();

    /**
     * @brief 查询客户端当前是否处于已连接状态。
     *
     * @return 已连接返回 true，否则返回 false。
     */
    bool IsConnected() const;

    /**
     * @brief 确保指定持久路径及其父路径存在。
     *
     * @param path 需要确保存在的持久节点路径。
     * @return 路径检查或创建成功返回 true，否则返回 false。
     */
    bool EnsurePath(const std::string& path);

    /**
     * @brief 创建一个临时节点。
     *
     * @param path 要创建的临时节点完整路径。
     * @param value 节点数据内容。
     * @return 创建成功或节点已存在时返回 true，否则返回 false。
     */
    bool CreateEphemeral(const std::string& path, const std::string& value);

private:
    /**
     * @brief ZooKeeper 全局 watcher 静态回调入口。
     *
     * @param zh ZooKeeper 连接句柄。
     * @param type 事件类型。
     * @param state 连接状态。
     * @param path 与事件相关的路径。
     * @param watcher_ctx 在初始化时传入的用户上下文。
     */
    static void GlobalWatcher(zhandle_t* zh, int type, int state, const char* path, void* watcher_ctx);

    /**
     * @brief 处理 watcher 回调并更新连接状态。
     *
     * @param type 事件类型。
     * @param state 连接状态。
     * @param path 与事件相关的路径。
     */
    void HandleWatcher(int type, int state, const char* path);

    /**
     * @brief 创建单个持久节点。
     *
     * @param path 要创建的持久节点路径。
     * @return 创建成功或节点已存在时返回 true，否则返回 false。
     */
    bool CreatePersistentNode(const std::string& path);

    zhandle_t* handle_;
    mutable std::mutex mutex_;
    std::condition_variable connected_cv_;
    bool connected_;
};
