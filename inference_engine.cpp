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

    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Load model
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

    input_node_names_ = {input_name_.c_str()};
    output_node_names_ = {output_name_.c_str()};

    std::cout << "ONNX model loaded successfully: " << model_path << std::endl;
}

InferenceEngine::~InferenceEngine() = default;

std::vector<float> InferenceEngine::Preprocess(const std::vector<uint8_t>& image_bytes) {
    int width, height, channels;
    // Load image from memory
    unsigned char* img_data = stbi_load_from_memory(
        image_bytes.data(), static_cast<int>(image_bytes.size()),
        &width, &height, &channels, 3); // Force 3 channels (RGB)

    if (!img_data) {
        throw std::runtime_error("Failed to decode image");
    }

    // Resize to 224x224
    const int target_w = 224;
    const int target_h = 224;
    const int target_c = 3;
    unsigned char* resized_data = new unsigned char[target_w * target_h * target_c];

    // Use STBIR_FILTER_TRIANGLE (Bilinear) to match PIL's Bilinear more closely
    // Actually, stbir v2 default API might not expose this easily without extended.
    // Let's check if there is an extended function.
    // Wait, the API for extended in v2 is stbir_resize().
    // Let's use the full resize setup.
    STBIR_RESIZE resize;
    stbir_resize_init(&resize, img_data, width, height, 0, resized_data, target_w, target_h, 0, (stbir_pixel_layout)target_c, STBIR_TYPE_UINT8);
    stbir_set_filters(&resize, STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE);
    stbir_resize_extended(&resize);

    stbi_image_free(img_data);

    // Normalize and convert to NCHW
    // ResNet18 uses ImageNet mean/std
    std::vector<float> tensor_data(target_w * target_h * target_c);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < target_c; ++c) {
        for (int y = 0; y < target_h; ++y) {
            for (int x = 0; x < target_w; ++x) {
                int hw_idx = y * target_w + x;
                int nhwc_idx = c * (target_h * target_w) + hw_idx; // NCHW layout (N=1)
                int hwc_idx = hw_idx * target_c + c;               // HWC layout (from stb)

                float pixel = resized_data[hwc_idx] / 255.0f;
                tensor_data[nhwc_idx] = (pixel - mean[c]) / std_dev[c];
            }
        }
    }

    delete[] resized_data;
    return tensor_data;
}

std::pair<int, float> InferenceEngine::Predict(const std::vector<uint8_t>& image_bytes) {
    // 1. Preprocess
    std::vector<float> input_tensor_values = Preprocess(image_bytes);

    // 2. Prepare inputs
    std::vector<int64_t> input_shape = {1, 3, 224, 224}; // Batch size 1

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 3. Run inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input_tensor, 1,
        output_node_names_.data(), 1
    );

    // 4. Postprocess (find argmax)
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    size_t output_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    // Softmax
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
