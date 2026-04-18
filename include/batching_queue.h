#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "inference_engine.h"

// 推理请求结构体，包含原始图像字节与异步返回结果所需的 promise。
struct InferenceRequest {
    std::vector<uint8_t> image_bytes;
    std::promise<std::pair<int, float>> promise;
};

// 动态批处理队列，负责聚合并发请求并触发批量推理。
class BatchingQueue {
public:
    // max_batch_size：单个批次的最大请求数。
    // max_latency_ms：首个请求允许等待的最大时延，单位为毫秒。
    // max_queue_size：队列允许缓存的最大请求数，超过后直接拒绝新请求。
    BatchingQueue(std::shared_ptr<InferenceEngine> engine,
                  size_t max_batch_size,
                  int max_latency_ms,
                  size_t max_queue_size = 4096);
    ~BatchingQueue();

    BatchingQueue(const BatchingQueue&) = delete;
    BatchingQueue& operator=(const BatchingQueue&) = delete;

    // 提交一个推理请求，返回 future 供调用方异步获取结果。
    std::future<std::pair<int, float>> Submit(std::vector<uint8_t> image_bytes);

private:
    void WorkerThread();

    std::shared_ptr<InferenceEngine> engine_;
    size_t max_batch_size_;
    std::chrono::milliseconds max_latency_;
    size_t max_queue_size_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<InferenceRequest>> queue_;
    std::atomic<bool> stop_;
    std::thread worker_thread_;
};
