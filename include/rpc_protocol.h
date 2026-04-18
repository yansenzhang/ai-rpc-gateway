#pragma once

#include "inference.pb.h"
#include "muduo/net/protobuf/ProtobufCodecLite.h"

namespace inference {

// 自定义 TCP RPC 协议的 4 字节标签，用于 Muduo ProtobufCodecLite 帧校验。
extern const char kGatewayRpcTag[];

using GatewayRpcCodec = muduo::net::ProtobufCodecLiteT<GatewayRpcMessage, kGatewayRpcTag>;
using GatewayRpcMessagePtr = GatewayRpcCodec::ConcreteMessagePtr;

}  // namespace inference
