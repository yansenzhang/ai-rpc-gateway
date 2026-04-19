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

/**
 * @brief 表示一个待执行的推理请求。
 *
 * 请求对象保存原始图像数据以及与调用方同步结果所需的 promise，供批处理
 * 工作线程在完成推理后回填结果或异常。
 */
struct InferenceRequest {
    std::vector<uint8_t> image_bytes;
    std::promise<std::pair<int, float>> promise;
};

/**
 * @brief 面向并发请求的动态批处理队列。
 *
 * 该类负责在后台线程中聚合短时间窗口内到达的推理请求，并在达到最大批次
 * 或等待超时后统一调用 `InferenceEngine::PredictBatch`。调用方通过 future
 * 异步获取结果，从而避免在接入层直接阻塞。
 */
class BatchingQueue {
public:
    /**
     * @brief 创建一个带后台工作线程的批处理队列。
     *
     * @param engine 共享推理引擎实例。
     * @param max_batch_size 单个批次允许聚合的最大请求数。
     * @param max_latency_ms 首个请求允许等待后续请求拼批的最大时延，单位为毫秒。
     * @param max_queue_size 队列允许缓存的最大请求数，超过后直接拒绝新请求。
     */
    BatchingQueue(std::shared_ptr<InferenceEngine> engine,
                  size_t max_batch_size,
                  int max_latency_ms,
                  size_t max_queue_size = 4096);

    /**
     * @brief 停止后台工作线程并释放队列资源。
     */
    ~BatchingQueue();

    BatchingQueue(const BatchingQueue&) = delete;
    BatchingQueue& operator=(const BatchingQueue&) = delete;

    /**
     * @brief 提交一个推理请求并返回异步结果句柄。
     *
     * @param image_bytes 待推理图像的原始字节流。
     * @return 用于等待分类结果的 future。
     *
     * @note 该接口可被多个线程并发调用，内部使用互斥锁保护队列状态。
     */
    std::future<std::pair<int, float>> Submit(std::vector<uint8_t> image_bytes);

private:
    /**
     * @brief 后台工作线程主体。
     *
     * 工作线程负责等待新请求、按时间窗口聚合批次、执行批量推理，并将结果
     * 或异常逐一传播回对应请求的 promise。
     */
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
