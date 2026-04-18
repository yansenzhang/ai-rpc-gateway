#pragma once

#include <memory>

#include <muduo/net/EventLoop.h>
#include <muduo/net/InetAddress.h>
#include <muduo/net/TcpServer.h>

#include "inference_service_impl.h"
#include "rpc_protocol.h"

class RpcServer {
public:
    // loop：Muduo 事件循环对象。
    // listen_addr：服务监听地址。
    // service：业务处理实现。
    RpcServer(muduo::net::EventLoop* loop,
              const muduo::net::InetAddress& listen_addr,
              std::shared_ptr<InferenceServiceImpl> service);

    // num_threads：Muduo I/O 线程数量。
    void SetThreadNum(int num_threads);

    void Start();

private:
    void OnConnection(const muduo::net::TcpConnectionPtr& conn);
    void OnRpcMessage(const muduo::net::TcpConnectionPtr& conn,
                      const inference::GatewayRpcMessagePtr& message,
                      muduo::Timestamp receive_time);
    void OnRpcError(const muduo::net::TcpConnectionPtr& conn,
                    muduo::net::Buffer* buf,
                    muduo::Timestamp receive_time,
                    muduo::net::ProtobufCodecLite::ErrorCode error_code);
    void SendResponse(const muduo::net::TcpConnectionPtr& conn,
                      const inference::GatewayRpcMessage& response);

    muduo::net::EventLoop* loop_;
    muduo::net::TcpServer server_;
    inference::GatewayRpcCodec codec_;
    std::shared_ptr<InferenceServiceImpl> service_;
};
