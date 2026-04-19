#pragma once

#include <functional>
#include <memory>
#include <string>

#include <muduo/base/ThreadPool.h>
#include <muduo/net/Callbacks.h>

#include "batching_queue.h"
#include "rpc_protocol.h"

/**
 * @brief RPC 业务处理层实现。
 *
 * 该类负责承接已经完成协议解码的 RPC 请求，完成参数校验、超时控制、
 * 批处理队列投递以及响应封装。实际等待 future 与发送响应的动作会被转移到
 * 后台线程池执行，以避免阻塞 Muduo 的 I/O 线程。
 */
class InferenceServiceImpl {
public:
    using SendResponseCallback = std::function<void(const muduo::net::TcpConnectionPtr&, const inference::GatewayRpcMessage&)>;

    /**
     * @brief 构造业务处理层。
     *
     * @param queue 共享批处理队列实例。
     * @param worker_threads 用于等待推理结果并回传响应的后台线程数量。
     */
    InferenceServiceImpl(std::shared_ptr<BatchingQueue> queue, int worker_threads);

    /**
     * @brief 停止内部后台线程池。
     */
    ~InferenceServiceImpl();

    /**
     * @brief 处理一个已解码的 RPC 信封消息。
     *
     * @param conn 当前请求对应的 TCP 连接。
     * @param request_message 已经通过协议层解码完成的 RPC 信封消息。
     * @param send_response 用于异步发送响应消息的回调。
     */
    void HandleMessage(const muduo::net::TcpConnectionPtr& conn,
                       const inference::GatewayRpcMessage& request_message,
                       const SendResponseCallback& send_response);

private:
    /**
     * @brief 获取当前系统时间戳，单位为毫秒。
     */
    static int64_t CurrentTimeMs();

    /**
     * @brief 从请求消息中提取业务请求 ID。
     *
     * 若业务层未显式提供 `request_id`，则回退为 `correlation_id` 的字符串形式。
     *
     * @param request_message 已解码的 RPC 请求消息。
     * @return 可用于链路追踪的请求 ID。
     */
    static std::string BuildRequestId(const inference::GatewayRpcMessage& request_message);

    /**
     * @brief 处理 Predict 业务请求。
     *
     * @param conn 当前请求对应的 TCP 连接。
     * @param correlation_id 请求与响应共享的关联 ID。
     * @param request 已解析出的业务层推理请求。
     * @param send_response 用于发送响应消息的网络回调。
     */
    void ProcessPredictRequest(const muduo::net::TcpConnectionPtr& conn,
                               uint64_t correlation_id,
                               inference::PredictRequest request,
                               const SendResponseCallback& send_response);

    /**
     * @brief 构造统一格式的失败响应。
     *
     * @param correlation_id 请求与响应共享的关联 ID。
     * @param request_id 业务请求 ID。
     * @param status_code 业务状态码。
     * @param error_message 返回给客户端的错误原因。
     * @return 封装完成的 RPC 响应消息。
     */
    inference::GatewayRpcMessage BuildErrorResponse(uint64_t correlation_id,
                                                    const std::string& request_id,
                                                    inference::PredictStatusCode status_code,
                                                    const std::string& error_message) const;

    std::shared_ptr<BatchingQueue> queue_;
    muduo::ThreadPool worker_pool_;
};
