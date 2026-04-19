#include "rpc_server.h"

#include <iostream>

#include <muduo/base/Logging.h>
#include <muduo/net/Buffer.h>

RpcServer::RpcServer(muduo::net::EventLoop* loop,
                     const muduo::net::InetAddress& listen_addr,
                     std::shared_ptr<InferenceServiceImpl> service)
    : loop_(loop),
      server_(loop, listen_addr, "AIInferenceGateway"),
      codec_(std::bind(&RpcServer::OnRpcMessage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
             {},
             std::bind(&RpcServer::OnRpcError, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)),
      service_(std::move(service)) {
    // 连接事件与消息事件分别交由服务端回调和协议编解码器处理。
    server_.setConnectionCallback(std::bind(&RpcServer::OnConnection, this, std::placeholders::_1));
    server_.setMessageCallback(std::bind(&inference::GatewayRpcCodec::onMessage, &codec_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

void RpcServer::SetThreadNum(int num_threads) {
    server_.setThreadNum(num_threads);
}

void RpcServer::Start() {
    server_.start();
}

void RpcServer::OnConnection(const muduo::net::TcpConnectionPtr& conn) {
    if (conn->connected()) {
        std::cout << "[RpcServer] New connection from " << conn->peerAddress().toIpPort() << std::endl;
    } else {
        std::cout << "[RpcServer] Connection closed: " << conn->peerAddress().toIpPort() << std::endl;
    }
}

void RpcServer::OnRpcMessage(const muduo::net::TcpConnectionPtr& conn,
                             const inference::GatewayRpcMessagePtr& message,
                             muduo::Timestamp receive_time) {
    (void)receive_time;

    // 编解码成功后，直接把业务消息转交给服务层处理。
    service_->HandleMessage(conn,
                            *message,
                            std::bind(&RpcServer::SendResponse, this, std::placeholders::_1, std::placeholders::_2));
}

void RpcServer::OnRpcError(const muduo::net::TcpConnectionPtr& conn,
                           muduo::net::Buffer* buf,
                           muduo::Timestamp receive_time,
                           muduo::net::ProtobufCodecLite::ErrorCode error_code) {
    (void)buf;
    (void)receive_time;

    // 协议解析失败通常说明对端发送了损坏或不兼容的数据，主动关闭连接以避免继续污染会话状态。
    std::cerr << "[RpcServer] Failed to parse incoming message, error="
              << muduo::net::ProtobufCodecLite::errorCodeToString(error_code)
              << ", peer=" << conn->peerAddress().toIpPort() << std::endl;
    conn->shutdown();
}

void RpcServer::SendResponse(const muduo::net::TcpConnectionPtr& conn,
                             const inference::GatewayRpcMessage& response) {
    if (!conn || !conn->connected()) {
        return;
    }
    codec_.send(conn, response);
}
