#include "batching_queue.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

BatchingQueue::BatchingQueue(std::shared_ptr<InferenceEngine> engine,
                             size_t max_batch_size,
                             int max_latency_ms,
                             size_t max_queue_size)
    : engine_(std::move(engine)),
      max_batch_size_(max_batch_size),
      max_latency_(max_latency_ms),
      max_queue_size_(max_queue_size),
      stop_(false) {
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
        if (stop_) {
            throw std::runtime_error("BatchingQueue is stopped");
        }
        if (queue_.size() >= max_queue_size_) {
            throw std::runtime_error("BatchingQueue is full");
        }
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

            // 等待队列中出现新请求，或等待退出信号。
            cv_.wait(lock, [this] { return !queue_.empty() || stop_; });

            if (stop_ && queue_.empty()) {
                break;
            }

            // 以当前首个请求开始计时，在“凑满批次”与“达到最大等待时延”之间择一触发。
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
            queue_.erase(queue_.begin(), queue_.begin() + static_cast<long>(batch_size));
        }

        if (batch.empty()) {
            continue;
        }

        try {
            std::cout << "[BatchingQueue] Running batch inference, batch size: " << batch.size() << std::endl;

            // 把批次中的字节流提取出来，交给推理引擎执行一次批量推理。
            std::vector<std::vector<uint8_t>> batch_images;
            batch_images.reserve(batch.size());
            for (const auto& req : batch) {
                batch_images.push_back(req->image_bytes);
            }

            auto results = engine_->PredictBatch(batch_images);
            if (results.size() != batch.size()) {
                throw std::runtime_error("PredictBatch returned mismatched result count");
            }

            // 推理成功后，按请求顺序逐一回填结果。
            for (size_t i = 0; i < batch.size(); ++i) {
                batch[i]->promise.set_value(results[i]);
            }
        } catch (...) {
            // 若整批推理失败，则把同一个异常传播给该批次中的所有等待方。
            for (auto& req : batch) {
                req->promise.set_exception(std::current_exception());
            }
        }
    }
}
