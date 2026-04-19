#pragma once

#include <memory>

#include <muduo/net/EventLoop.h>
#include <muduo/net/InetAddress.h>
#include <muduo/net/TcpServer.h>

#include "inference_service_impl.h"
#include "rpc_protocol.h"

/**
 * @brief 基于 Muduo 的 RPC 接入服务端。
 *
 * 该类封装 TCP 监听、协议解码、连接生命周期日志以及业务层回调转发，
 * 负责把网络层收到的消息交给 `InferenceServiceImpl` 处理，并将业务响应写回连接。
 */
class RpcServer {
public:
    /**
     * @brief 构造 RPC 服务端。
     *
     * @param loop Muduo 事件循环对象。
     * @param listen_addr 服务监听地址。
     * @param service 负责处理业务消息的服务实现。
     */
    RpcServer(muduo::net::EventLoop* loop,
              const muduo::net::InetAddress& listen_addr,
              std::shared_ptr<InferenceServiceImpl> service);

    /**
     * @brief 设置 Muduo I/O 线程数量。
     *
     * @param num_threads I/O 线程数量。
     */
    void SetThreadNum(int num_threads);

    /**
     * @brief 启动 TCP 服务监听。
     */
    void Start();

private:
    /**
     * @brief 处理连接建立与断开事件。
     *
     * @param conn 当前变化的连接对象。
     */
    void OnConnection(const muduo::net::TcpConnectionPtr& conn);

    /**
     * @brief 处理一条成功解码的 RPC 消息。
     *
     * @param conn 消息所属连接。
     * @param message 已解码的 RPC 消息对象。
     * @param receive_time Muduo 提供的接收时间戳。
     */
    void OnRpcMessage(const muduo::net::TcpConnectionPtr& conn,
                      const inference::GatewayRpcMessagePtr& message,
                      muduo::Timestamp receive_time);

    /**
     * @brief 处理协议解码失败场景。
     *
     * @param conn 当前连接。
     * @param buf 尚未完成解析的输入缓冲区。
     * @param receive_time Muduo 提供的接收时间戳。
     * @param error_code 协议编解码错误码。
     */
    void OnRpcError(const muduo::net::TcpConnectionPtr& conn,
                    muduo::net::Buffer* buf,
                    muduo::Timestamp receive_time,
                    muduo::net::ProtobufCodecLite::ErrorCode error_code);

    /**
     * @brief 将业务响应编码后写回客户端。
     *
     * @param conn 目标连接。
     * @param response 待发送的响应消息。
     */
    void SendResponse(const muduo::net::TcpConnectionPtr& conn,
                      const inference::GatewayRpcMessage& response);

    muduo::net::EventLoop* loop_;
    muduo::net::TcpServer server_;
    inference::GatewayRpcCodec codec_;
    std::shared_ptr<InferenceServiceImpl> service_;
};
