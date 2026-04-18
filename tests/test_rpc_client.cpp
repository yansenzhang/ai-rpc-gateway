#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <string>

#include <muduo/net/Buffer.h>
#include <muduo/net/EventLoop.h>
#include <muduo/net/EventLoopThread.h>
#include <muduo/net/InetAddress.h>
#include <muduo/net/TcpClient.h>

#include "rpc_protocol.h"

namespace {

std::vector<uint8_t> ReadFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    throw std::runtime_error("Error reading file: " + filepath);
}

class RpcClient {
public:
    // server_addr：服务端监听地址。
    RpcClient(muduo::net::EventLoop* loop, const muduo::net::InetAddress& server_addr)
        : client_(loop, server_addr, "AIInferenceGatewayClient"),
          codec_(std::bind(&RpcClient::OnRpcMessage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)),
          next_correlation_id_(1),
          received_(false) {
        client_.setConnectionCallback(std::bind(&RpcClient::OnConnection, this, std::placeholders::_1));
        client_.setMessageCallback(std::bind(&inference::GatewayRpcCodec::onMessage, &codec_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    }

    void Connect() {
        client_.connect();
    }

    // image_bytes：待发送的图像原始字节。
    inference::PredictResponse Predict(const std::vector<uint8_t>& image_bytes) {
        std::unique_lock<std::mutex> lock(mutex_);
        connected_cv_.wait(lock, [this] { return static_cast<bool>(connection_) && connection_->connected(); });

        const uint64_t correlation_id = next_correlation_id_++;
        inference::GatewayRpcMessage request;
        request.set_message_type(inference::GATEWAY_MESSAGE_TYPE_REQUEST);
        request.set_correlation_id(correlation_id);
        request.set_method_name("Predict");

        auto* predict_request = request.mutable_predict_request();
        predict_request->set_request_id(std::to_string(correlation_id));
        predict_request->set_image_format("jpeg");
        predict_request->set_timeout_ms(3000);
        predict_request->set_image_bytes(image_bytes.data(), static_cast<int>(image_bytes.size()));

        received_ = false;
        codec_.send(connection_, request);

        response_cv_.wait(lock, [this] { return received_; });
        return response_;
    }

private:
    void OnConnection(const muduo::net::TcpConnectionPtr& conn) {
        std::lock_guard<std::mutex> lock(mutex_);
        connection_ = conn;
        if (conn->connected()) {
            connected_cv_.notify_all();
        }
    }

    void OnRpcMessage(const muduo::net::TcpConnectionPtr& conn,
                      const inference::GatewayRpcMessagePtr& message,
                      muduo::Timestamp receive_time) {
        (void)conn;
        (void)receive_time;

        if (message->message_type() != inference::GATEWAY_MESSAGE_TYPE_RESPONSE || !message->has_predict_response()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        response_ = message->predict_response();
        received_ = true;
        response_cv_.notify_all();
    }

    muduo::net::TcpClient client_;
    inference::GatewayRpcCodec codec_;
    std::mutex mutex_;
    std::condition_variable connected_cv_;
    std::condition_variable response_cv_;
    muduo::net::TcpConnectionPtr connection_;
    uint64_t next_correlation_id_;
    inference::PredictResponse response_;
    bool received_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <server_ip> <server_port> <model_path_unused> <image_path>" << std::endl;
        return 1;
    }

    const std::string server_ip = argv[1];
    const uint16_t server_port = static_cast<uint16_t>(std::stoi(argv[2]));
    const std::string image_path = argv[4];

    try {
        auto image_bytes = ReadFile(image_path);

        muduo::net::EventLoopThread loop_thread;
        muduo::net::EventLoop* loop = loop_thread.startLoop();
        muduo::net::InetAddress server_addr(server_ip, server_port);
        RpcClient client(loop, server_addr);

        client.Connect();
        auto response = client.Predict(image_bytes);

        std::cout << "Status: " << response.status_code() << std::endl;
        std::cout << "Error: " << response.error_message() << std::endl;
        std::cout << "Class ID: " << response.class_id() << std::endl;
        std::cout << "Confidence: " << response.confidence() << std::endl;

        loop->runInLoop([loop]() {
            loop->quit();
        });
    } catch (const std::exception& e) {
        std::cerr << "Client error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
