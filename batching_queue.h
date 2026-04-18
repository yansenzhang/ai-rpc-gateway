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

struct InferenceRequest {
    std::vector<uint8_t> image_bytes;
    std::promise<std::pair<int, float>> promise;
};

class BatchingQueue {
public:
    BatchingQueue(std::shared_ptr<InferenceEngine> engine, size_t max_batch_size, int max_latency_ms);
    ~BatchingQueue();

    BatchingQueue(const BatchingQueue&) = delete;
    BatchingQueue& operator=(const BatchingQueue&) = delete;

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
