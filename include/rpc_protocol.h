#pragma once

#include "inference.pb.h"
#include "muduo/net/protobuf/ProtobufCodecLite.h"

namespace inference {

/**
 * @brief 自定义 TCP RPC 协议的 4 字节帧标签。
 *
 * 该标签用于配合 Muduo `ProtobufCodecLite` 做帧边界与协议合法性校验，
 * 客户端与服务端必须保持一致。
 */
extern const char kGatewayRpcTag[];

using GatewayRpcCodec = muduo::net::ProtobufCodecLiteT<GatewayRpcMessage, kGatewayRpcTag>;
using GatewayRpcMessagePtr = GatewayRpcCodec::ConcreteMessagePtr;

}  // namespace inference
