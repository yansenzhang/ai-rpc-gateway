#include "inference_service_impl.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>

InferenceServiceImpl::InferenceServiceImpl(std::shared_ptr<BatchingQueue> queue, int worker_threads)
    : queue_(std::move(queue)), worker_pool_("InferenceServiceWorkerPool") {
    worker_pool_.start(worker_threads);
}

InferenceServiceImpl::~InferenceServiceImpl() {
    worker_pool_.stop();
}

void InferenceServiceImpl::HandleMessage(const muduo::net::TcpConnectionPtr& conn,
                                         const inference::GatewayRpcMessage& request_message,
                                         const SendResponseCallback& send_response) {
    // 仅接受请求方向的信封消息，其他类型直接视为协议使用错误。
    if (request_message.message_type() != inference::GATEWAY_MESSAGE_TYPE_REQUEST) {
        auto response = BuildErrorResponse(request_message.correlation_id(),
                                           BuildRequestId(request_message),
                                           inference::PREDICT_STATUS_INVALID_REQUEST,
                                           "Only request messages are accepted");
        send_response(conn, response);
        return;
    }

    // 当前服务仅暴露 Predict 方法，避免不受支持的方法进入业务处理流程。
    if (request_message.method_name() != "Predict") {
        auto response = BuildErrorResponse(request_message.correlation_id(),
                                           BuildRequestId(request_message),
                                           inference::PREDICT_STATUS_INVALID_REQUEST,
                                           "Unsupported RPC method");
        send_response(conn, response);
        return;
    }

    // Predict 请求必须携带业务层 payload，否则无法执行实际推理。
    if (!request_message.has_predict_request()) {
        auto response = BuildErrorResponse(request_message.correlation_id(),
                                           BuildRequestId(request_message),
                                           inference::PREDICT_STATUS_INVALID_REQUEST,
                                           "Predict request payload is missing");
        send_response(conn, response);
        return;
    }

    ProcessPredictRequest(conn,
                          request_message.correlation_id(),
                          request_message.predict_request(),
                          send_response);
}

int64_t InferenceServiceImpl::CurrentTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string InferenceServiceImpl::BuildRequestId(const inference::GatewayRpcMessage& request_message) {
    if (request_message.has_predict_request() && !request_message.predict_request().request_id().empty()) {
        return request_message.predict_request().request_id();
    }
    return std::to_string(request_message.correlation_id());
}

void InferenceServiceImpl::ProcessPredictRequest(const muduo::net::TcpConnectionPtr& conn,
                                                 uint64_t correlation_id,
                                                 inference::PredictRequest request,
                                                 const SendResponseCallback& send_response) {
    const std::string request_id = request.request_id().empty()
                                       ? std::to_string(correlation_id)
                                       : request.request_id();

    // 空图像请求不具备推理意义，直接返回参数错误。
    if (request.image_bytes().empty()) {
        auto response = BuildErrorResponse(correlation_id,
                                           request_id,
                                           inference::PREDICT_STATUS_INVALID_REQUEST,
                                           "image_bytes must not be empty");
        send_response(conn, response);
        return;
    }

    // 对单请求负载设置上限，避免异常大包长期占用内存与工作线程。
    constexpr size_t kMaxImageBytes = 10 * 1024 * 1024;
    if (request.image_bytes().size() > kMaxImageBytes) {
        auto response = BuildErrorResponse(correlation_id,
                                           request_id,
                                           inference::PREDICT_STATUS_INVALID_REQUEST,
                                           "image_bytes exceeds the 10MB limit");
        send_response(conn, response);
        return;
    }

    std::vector<uint8_t> image_bytes(request.image_bytes().begin(), request.image_bytes().end());
    const uint32_t timeout_ms = request.timeout_ms();
    const int64_t server_start_ts_ms = CurrentTimeMs();

    // 将等待批处理 future 的动作转移到业务线程池，避免阻塞 Muduo I/O 线程。
    worker_pool_.run([this, conn, correlation_id, request_id, image_bytes = std::move(image_bytes), timeout_ms, server_start_ts_ms, send_response]() mutable {
        try {
            auto future = queue_->Submit(std::move(image_bytes));

            std::pair<int, float> result;
            if (timeout_ms > 0) {
                auto wait_result = future.wait_for(std::chrono::milliseconds(timeout_ms));
                if (wait_result != std::future_status::ready) {
                    auto response = BuildErrorResponse(correlation_id,
                                                       request_id,
                                                       inference::PREDICT_STATUS_TIMEOUT,
                                                       "Inference request timed out");
                    response.mutable_predict_response()->set_server_start_ts_ms(server_start_ts_ms);
                    response.mutable_predict_response()->set_server_end_ts_ms(CurrentTimeMs());
                    send_response(conn, response);
                    return;
                }
            }
            result = future.get();

            inference::GatewayRpcMessage response;
            response.set_message_type(inference::GATEWAY_MESSAGE_TYPE_RESPONSE);
            response.set_correlation_id(correlation_id);
            response.set_method_name("Predict");

            auto* predict_response = response.mutable_predict_response();
            predict_response->set_status_code(inference::PREDICT_STATUS_OK);
            predict_response->set_error_message("");
            predict_response->set_class_id(result.first);
            predict_response->set_confidence(result.second);
            predict_response->set_request_id(request_id);
            predict_response->set_server_start_ts_ms(server_start_ts_ms);
            predict_response->set_server_end_ts_ms(CurrentTimeMs());

            send_response(conn, response);
        } catch (const std::exception& e) {
            // 队列满属于可预期业务状态，其余异常统一映射为内部错误。
            auto response = BuildErrorResponse(correlation_id,
                                               request_id,
                                               std::string(e.what()) == "BatchingQueue is full"
                                                   ? inference::PREDICT_STATUS_QUEUE_FULL
                                                   : inference::PREDICT_STATUS_INTERNAL_ERROR,
                                               e.what());
            response.mutable_predict_response()->set_server_start_ts_ms(server_start_ts_ms);
            response.mutable_predict_response()->set_server_end_ts_ms(CurrentTimeMs());
            send_response(conn, response);
        }
    });
}

inference::GatewayRpcMessage InferenceServiceImpl::BuildErrorResponse(uint64_t correlation_id,
                                                                      const std::string& request_id,
                                                                      inference::PredictStatusCode status_code,
                                                                      const std::string& error_message) const {
    inference::GatewayRpcMessage response;
    response.set_message_type(inference::GATEWAY_MESSAGE_TYPE_RESPONSE);
    response.set_correlation_id(correlation_id);
    response.set_method_name("Predict");

    auto* predict_response = response.mutable_predict_response();
    predict_response->set_status_code(status_code);
    predict_response->set_error_message(error_message);
    predict_response->set_request_id(request_id);
    predict_response->set_server_end_ts_ms(CurrentTimeMs());
    return response;
}
