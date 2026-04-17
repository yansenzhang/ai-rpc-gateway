#include "inference_engine.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

InferenceEngine::InferenceEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "AI-Gateway-Env"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

    // 创建会话配置选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 加载模型
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

    input_node_names_ = {input_name_.c_str()};
    output_node_names_ = {output_name_.c_str()};

    std::cout << "ONNX model loaded successfully: " << model_path << std::endl;
}

InferenceEngine::~InferenceEngine() = default;

std::vector<float> InferenceEngine::Preprocess(const std::vector<uint8_t>& image_bytes) {
    int width, height, channels;
    // 从内存中加载图像
    unsigned char* img_data = stbi_load_from_memory(
        image_bytes.data(), static_cast<int>(image_bytes.size()),
        &width, &height, &channels, 3); // 强制使用 3 通道 (RGB)

    if (!img_data) {
        throw std::runtime_error("Failed to decode image");
    }

    // 缩放至 224x224 分辨率
    const int target_w = 224;
    const int target_h = 224;
    const int target_c = 3;
    unsigned char* resized_data = new unsigned char[target_w * target_h * target_c];

    // 使用 STBIR_FILTER_TRIANGLE (双线性插值) 滤波器，尽可能对齐 PIL 的双线性插值逻辑
    STBIR_RESIZE resize;
    stbir_resize_init(&resize, img_data, width, height, 0, resized_data, target_w, target_h, 0, (stbir_pixel_layout)target_c, STBIR_TYPE_UINT8);
    stbir_set_filters(&resize, STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE);
    stbir_resize_extended(&resize);

    stbi_image_free(img_data);

    // 归一化并转换为 NCHW 内存布局
    // ResNet18 使用 ImageNet 数据集的均值与标准差
    std::vector<float> tensor_data(target_w * target_h * target_c);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < target_c; ++c) {
        for (int y = 0; y < target_h; ++y) {
            for (int x = 0; x < target_w; ++x) {
                int hw_idx = y * target_w + x;
                int nhwc_idx = c * (target_h * target_w) + hw_idx; // NCHW 内存布局 (N=1)
                int hwc_idx = hw_idx * target_c + c;               // HWC 内存布局 (来自 stb 库)

                float pixel = resized_data[hwc_idx] / 255.0f;
                tensor_data[nhwc_idx] = (pixel - mean[c]) / std_dev[c];
            }
        }
    }

    delete[] resized_data;
    return tensor_data;
}

std::pair<int, float> InferenceEngine::Predict(const std::vector<uint8_t>& image_bytes) {
    // 1. 预处理
    std::vector<float> input_tensor_values = Preprocess(image_bytes);

    // 2. 准备输入张量
    std::vector<int64_t> input_shape = {1, 3, 224, 224}; // 批处理大小 (Batch size) 为 1

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 3. 执行推理
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input_tensor, 1,
        output_node_names_.data(), 1
    );

    // 4. 后处理 (寻找 argmax)
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    size_t output_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    // 计算 Softmax
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