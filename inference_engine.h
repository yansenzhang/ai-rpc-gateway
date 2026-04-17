#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <onnxruntime_cxx_api.h>

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path);
    ~InferenceEngine();

    // Prevent copying
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    // Single inference: takes raw image bytes (e.g. JPEG/PNG), returns (class_id, confidence)
    std::pair<int, float> Predict(const std::vector<uint8_t>& image_bytes);

private:
    std::vector<float> Preprocess(const std::vector<uint8_t>& image_bytes);

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    std::string input_name_ = "input";
    std::string output_name_ = "output";
};
