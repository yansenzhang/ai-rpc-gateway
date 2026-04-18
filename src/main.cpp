#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <muduo/net/EventLoop.h>
#include <muduo/net/InetAddress.h>

#include "inference_service_impl.h"
#include "rpc_server.h"
#include "service_registry.h"
#include "zk_client.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <listen_ip> <listen_port> [zk_hosts]" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string listen_ip = argv[2];
    const uint16_t listen_port = static_cast<uint16_t>(std::stoi(argv[3]));
    const std::string zk_hosts = argc >= 5 ? argv[4] : "";

    try {
        std::cout << "[Gateway] Initializing inference engine" << std::endl;
        auto engine = std::make_shared<InferenceEngine>(model_path);
        auto queue = std::make_shared<BatchingQueue>(engine, 16, 5, 4096);
        auto service = std::make_shared<InferenceServiceImpl>(queue, 4);

        muduo::net::EventLoop loop;
        muduo::net::InetAddress listen_addr(listen_ip, listen_port);
        RpcServer server(&loop, listen_addr, service);
        server.SetThreadNum(4);
        server.Start();

        std::shared_ptr<ZkClient> zk_client;
        std::unique_ptr<ServiceRegistry> service_registry;
        if (!zk_hosts.empty()) {
            zk_client = std::make_shared<ZkClient>();
            if (zk_client->Connect(zk_hosts, 3000)) {
                service_registry = std::make_unique<ServiceRegistry>(zk_client);
                const std::string endpoint = listen_ip + ":" + std::to_string(listen_port);
                const std::string metadata = std::string("{\"model\":\"") + model_path +
                                             "\",\"max_batch_size\":16,\"max_latency_ms\":5}";
                if (!service_registry->Register("inference.InferenceService", endpoint, metadata)) {
                    std::cerr << "[Gateway] Failed to register service to ZooKeeper" << std::endl;
                }
            } else {
                std::cerr << "[Gateway] ZooKeeper connection failed, RPC service will continue without registration" << std::endl;
            }
        }

        std::cout << "[Gateway] Listening on " << listen_ip << ":" << listen_port << std::endl;
        loop.loop();
    } catch (const std::exception& e) {
        std::cerr << "[Gateway] Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
