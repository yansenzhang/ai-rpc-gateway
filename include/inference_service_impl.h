#pragma once

#include <functional>
#include <memory>
#include <string>

#include <muduo/base/ThreadPool.h>
#include <muduo/net/Callbacks.h>

#include "batching_queue.h"
#include "rpc_protocol.h"

class InferenceServiceImpl {
public:
    using SendResponseCallback = std::function<void(const muduo::net::TcpConnectionPtr&, const inference::GatewayRpcMessage&)>;

    // queue：共享批处理队列。
    // worker_threads：用于等待 future 并回传响应的后台线程数。
    InferenceServiceImpl(std::shared_ptr<BatchingQueue> queue, int worker_threads);
    ~InferenceServiceImpl();

    // request_message：已完成解码的 RPC 信封消息。
    // send_response：用于异步回传响应的发送回调。
    void HandleMessage(const muduo::net::TcpConnectionPtr& conn,
                       const inference::GatewayRpcMessage& request_message,
                       const SendResponseCallback& send_response);

private:
    static int64_t CurrentTimeMs();
    static std::string BuildRequestId(const inference::GatewayRpcMessage& request_message);

    // correlation_id：请求与响应的关联 ID。
    // request：业务层推理请求。
    // send_response：用于发送响应的网络回调。
    void ProcessPredictRequest(const muduo::net::TcpConnectionPtr& conn,
                               uint64_t correlation_id,
                               inference::PredictRequest request,
                               const SendResponseCallback& send_response);

    // correlation_id：请求与响应的关联 ID。
    // request_id：业务请求 ID。
    // status_code：响应状态码。
    // error_message：失败时返回的错误原因。
    inference::GatewayRpcMessage BuildErrorResponse(uint64_t correlation_id,
                                                    const std::string& request_id,
                                                    inference::PredictStatusCode status_code,
                                                    const std::string& error_message) const;

    std::shared_ptr<BatchingQueue> queue_;
    muduo::ThreadPool worker_pool_;
};
