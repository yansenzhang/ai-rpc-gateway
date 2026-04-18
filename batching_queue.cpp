#include "batching_queue.h"
#include <iostream>
#include <chrono>

BatchingQueue::BatchingQueue(std::shared_ptr<InferenceEngine> engine, size_t max_batch_size, int max_latency_ms)
    : engine_(std::move(engine)), max_batch_size_(max_batch_size), max_latency_(max_latency_ms), stop_(false) {
    worker_thread_ = std::thread(&BatchingQueue::WorkerThread, this);
}

BatchingQueue::~BatchingQueue() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

std::future<std::pair<int, float>> BatchingQueue::Submit(std::vector<uint8_t> image_bytes) {
    auto req = std::make_unique<InferenceRequest>();
    req->image_bytes = std::move(image_bytes);
    auto future = req->promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(req));
    }
    cv_.notify_one();
    return future;
}

void BatchingQueue::WorkerThread() {
    while (true) {
        std::vector<std::unique_ptr<InferenceRequest>> batch;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            
            // Wait until queue is not empty or we need to stop
            cv_.wait(lock, [this] { return !queue_.empty() || stop_; });
            
            if (stop_ && queue_.empty()) {
                break;
            }
            
            // At this point, queue is NOT empty. We have our first request.
            // Wait for max_batch_size or max_latency timeout
            auto timeout_time = std::chrono::steady_clock::now() + max_latency_;
            cv_.wait_until(lock, timeout_time, [this] {
                return queue_.size() >= max_batch_size_ || stop_;
            });

            if (stop_ && queue_.empty()) {
                break;
            }

            size_t batch_size = std::min(queue_.size(), max_batch_size_);
            for (size_t i = 0; i < batch_size; ++i) {
                batch.push_back(std::move(queue_[i]));
            }
            queue_.erase(queue_.begin(), queue_.begin() + batch_size);
        }

        if (batch.empty()) {
            continue;
        }

        // 调用 InferenceEngine 进行批量推理
        try {
            // 目前 InferenceEngine 的 Predict 接口只支持单次推理
            // 所以先模拟逐个推理，后面需要修改 InferenceEngine 以支持真正的批量推理
            std::cout << "[WorkerThread] Executing batch inference for " << batch.size() << " requests." << std::endl;
            for (auto& req : batch) {
                auto result = engine_->Predict(req->image_bytes);
                req->promise.set_value(result);
            }
        } catch (...) {
            for (auto& req : batch) {
                try {
                    throw; // 重新抛出异常以设置给 promise
                } catch (...) {
                    req->promise.set_exception(std::current_exception());
                }
            }
        }
    }
}
