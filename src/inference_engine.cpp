#include "inference_engine.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

InferenceEngine::InferenceEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "AI-Gateway-Env"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    // 创建会话配置，并将算子内部并行度收敛到较小值，避免与上层批处理并发过度竞争 CPU。
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 加载 ONNX 模型并缓存输入输出节点名，后续推理可直接复用。
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    input_node_names_ = {input_name_.c_str()};
    output_node_names_ = {output_name_.c_str()};

    std::cout << "ONNX model loaded successfully: " << model_path << std::endl;
}

InferenceEngine::~InferenceEngine() = default;

std::vector<float> InferenceEngine::Preprocess(const std::vector<uint8_t>& image_bytes) {
    int width, height, channels;

    // 从内存字节流中解码图像，并强制转换为 3 通道 RGB 数据。
    unsigned char* img_data = stbi_load_from_memory(
        image_bytes.data(), static_cast<int>(image_bytes.size()),
        &width, &height, &channels, 3);

    if (!img_data) {
        throw std::runtime_error("Failed to decode image");
    }

    // 将输入图像缩放到模型要求的 224x224 分辨率。
    const int target_w = 224;
    const int target_h = 224;
    const int target_c = 3;
    unsigned char* resized_data = new unsigned char[target_w * target_h * target_c];

    // 使用三角滤波器实现双线性插值，使缩放效果尽量贴近常见图像预处理流程。
    STBIR_RESIZE resize;
    stbir_resize_init(&resize, img_data, width, height, 0, resized_data, target_w, target_h, 0, (stbir_pixel_layout)target_c, STBIR_TYPE_UINT8);
    stbir_set_filters(&resize, STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE);
    stbir_resize_extended(&resize);

    stbi_image_free(img_data);

    // 按 ResNet18 的输入要求做归一化，并把像素从 HWC 排列转换为 NCHW 排列。
    std::vector<float> tensor_data(target_w * target_h * target_c);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < target_c; ++c) {
        for (int y = 0; y < target_h; ++y) {
            for (int x = 0; x < target_w; ++x) {
                int hw_idx = y * target_w + x;
                int nchw_idx = c * (target_h * target_w) + hw_idx;
                int hwc_idx = hw_idx * target_c + c;

                float pixel = resized_data[hwc_idx] / 255.0f;
                tensor_data[nchw_idx] = (pixel - mean[c]) / std_dev[c];
            }
        }
    }

    delete[] resized_data;
    return tensor_data;
}

std::pair<int, float> InferenceEngine::Predict(const std::vector<uint8_t>& image_bytes) {
    // 先将单张图像预处理为模型输入张量。
    std::vector<float> input_tensor_values = Preprocess(image_bytes);

    // 单样本推理的批次维固定为 1。
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    // 执行一次前向推理，得到模型输出 logits。
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input_tensor, 1,
        output_node_names_.data(), 1);

    // 对输出执行 Softmax，并选取概率最大的类别作为最终结果。
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    size_t output_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    std::vector<float> probs(output_count);
    float max_val = *std::max_element(floatarr, floatarr + output_count);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < output_count; ++i) {
        probs[i] = std::exp(floatarr[i] - max_val);
        sum_exp += probs[i];
    }

    int best_class = 0;
    float best_prob = 0.0f;
    for (size_t i = 0; i < output_count; ++i) {
        probs[i] /= sum_exp;
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_class = static_cast<int>(i);
        }
    }

    return {best_class, best_prob};
}

std::vector<std::pair<int, float>> InferenceEngine::PredictBatch(const std::vector<std::vector<uint8_t>>& batch_image_bytes) {
    if (batch_image_bytes.empty()) {
        return {};
    }

    size_t batch_size = batch_image_bytes.size();
    size_t single_image_elements = 3 * 224 * 224;

    // 先逐张预处理，再把所有样本按批次顺序拼接成连续输入张量。
    std::vector<float> batched_tensor_values;
    batched_tensor_values.reserve(batch_size * single_image_elements);

    for (const auto& image_bytes : batch_image_bytes) {
        std::vector<float> tensor_data = Preprocess(image_bytes);
        batched_tensor_values.insert(batched_tensor_values.end(), tensor_data.begin(), tensor_data.end());
    }

    // 批量推理时仅调整批次维，其余输入形状与单样本保持一致。
    std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), 3, 224, 224};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        batched_tensor_values.data(),
        batched_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    // 一次性执行整批样本推理。
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input_tensor, 1,
        output_node_names_.data(), 1);

    // 逐个样本解析输出，并分别计算类别概率。
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    auto type_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    auto output_shape = type_info.GetShape();

    // 当前模型输出约定为 [batch_size, num_classes]。
    size_t num_classes = output_shape[1];

    std::vector<std::pair<int, float>> results;
    results.reserve(batch_size);

    for (size_t b = 0; b < batch_size; ++b) {
        float* sample_logits = floatarr + (b * num_classes);

        std::vector<float> probs(num_classes);
        float max_val = *std::max_element(sample_logits, sample_logits + num_classes);
        float sum_exp = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            probs[i] = std::exp(sample_logits[i] - max_val);
            sum_exp += probs[i];
        }

        int best_class = 0;
        float best_prob = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            probs[i] /= sum_exp;
            if (probs[i] > best_prob) {
                best_prob = probs[i];
                best_class = static_cast<int>(i);
            }
        }

        results.push_back({best_class, best_prob});
    }

    return results;
}
