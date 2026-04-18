#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include "inference_engine.h"

// 推理请求的结构体，包含输入图像数据和一个 promise，用于异步返回结果
struct InferenceRequest {
    std::vector<uint8_t> image_bytes;
    std::promise<std::pair<int, float>> promise;
};

// 动态批处理队列，负责收集并发请求并触发批量推理
class BatchingQueue {
public:
    BatchingQueue(std::shared_ptr<InferenceEngine> engine, size_t max_batch_size, int max_latency_ms);
    ~BatchingQueue();

    // 禁用拷贝构造和赋值操作
    BatchingQueue(const BatchingQueue&) = delete;
    BatchingQueue& operator=(const BatchingQueue&) = delete;

    // 提交一个推理请求，返回一个 future，用于异步获取结果
    std::future<std::pair<int, float>> Submit(std::vector<uint8_t> image_bytes);

private:
    void WorkerThread();

    std::shared_ptr<InferenceEngine> engine_;
    size_t max_batch_size_;
    std::chrono::milliseconds max_latency_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<InferenceRequest>> queue_;
    std::atomic<bool> stop_;
    std::thread worker_thread_;
};
