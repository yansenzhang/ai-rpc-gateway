#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <chrono>
#include <atomic>
#include "inference_engine.h"
#include "batching_queue.h"

std::vector<uint8_t> ReadFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (file.read((char*)buffer.data(), size)) {
        return buffer;
    }
    throw std::runtime_error("Error reading file: " + filepath);
}

std::atomic<int> success_count{0};

void Worker(std::shared_ptr<BatchingQueue> queue, const std::vector<uint8_t>& image_bytes, int id) {
    try {
        auto future = queue->Submit(image_bytes);
        auto result = future.get();
        if (result.first == 111) {
            success_count++;
        } else {
            std::cerr << "Thread " << id << " unexpected result: Class ID: " << result.first << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Thread " << id << " error: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    try {
        std::cout << "Initializing Engine..." << std::endl;
        auto engine = std::make_shared<InferenceEngine>(model_path);

        std::cout << "Initializing BatchingQueue..." << std::endl;
        auto queue = std::make_shared<BatchingQueue>(engine, 16, 5); // batch_size 16, latency 5ms

        std::cout << "Reading Image: " << image_path << std::endl;
        std::vector<uint8_t> image_bytes = ReadFile(image_path);

        std::cout << "Starting 100 threads for stress testing..." << std::endl;
        std::vector<std::thread> threads;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 100; ++i) {
            threads.push_back(std::thread(Worker, queue, image_bytes, i));
            // 极高频并发，基本不等待
        }

        for (auto& t : threads) {
            t.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "All threads finished in " << duration.count() << " ms." << std::endl;
        std::cout << "Successful predictions: " << success_count << " / 100" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Main error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
