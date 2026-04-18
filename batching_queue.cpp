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
            
            // 等待队列不为空，或收到停止信号
            cv_.wait(lock, [this] { return !queue_.empty() || stop_; });
            
            if (stop_ && queue_.empty()) {
                break;
            }
            
            // 此时队列不为空，开始计时等待凑齐 max_batch_size 或到达 max_latency
            auto timeout_time = std::chrono::steady_clock::now() + max_latency_;
            cv_.wait_until(lock, timeout_time, [this] {
                return queue_.size() >= max_batch_size_ || stop_;
            });

            if (stop_ && queue_.empty()) {
                break;
            }

            // 提取一个 batch 的请求
            size_t batch_size = std::min(queue_.size(), max_batch_size_);
            for (size_t i = 0; i < batch_size; ++i) {
                batch.push_back(std::move(queue_[i]));
            }
            queue_.erase(queue_.begin(), queue_.begin() + batch_size);
        }

        if (batch.empty()) {
            continue;
        }

        // 调用 InferenceEngine 进行推理
        try {
            std::cout << "[WorkerThread] 正在执行批量推理，请求数: " << batch.size() << std::endl;

            std::vector<std::vector<uint8_t>> batch_images;
            batch_images.reserve(batch.size());
            for (const auto& req : batch) {
                batch_images.push_back(req->image_bytes);
            }

            auto results = engine_->PredictBatch(batch_images);

            for (size_t i = 0; i < batch.size(); ++i) {
                batch[i]->promise.set_value(results[i]);
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
