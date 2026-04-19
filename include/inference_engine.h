#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

/**
 * @brief 基于 ONNX Runtime 的图像分类推理引擎。
 *
 * 该类负责模型加载、图像预处理、单样本推理与批量推理，向上层屏蔽
 * ONNX Runtime 会话管理与张量构造细节。对象在构造完成后持有稳定的模型
 * 会话与输入输出节点信息，供 RPC 服务和批处理队列复用。
 */
class InferenceEngine {
public:
    /**
     * @brief 使用指定模型路径初始化推理引擎。
     *
     * @param model_path ONNX 模型文件路径。
     */
    explicit InferenceEngine(const std::string& model_path);

    /**
     * @brief 析构推理引擎并释放 ONNX Runtime 相关资源。
     */
    ~InferenceEngine();

    /**
     * @brief 禁用拷贝构造，避免重复持有底层会话资源。
     */
    InferenceEngine(const InferenceEngine&) = delete;

    /**
     * @brief 禁用拷贝赋值，避免重复持有底层会话资源。
     */
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /**
     * @brief 对单张图像执行分类推理。
     *
     * @param image_bytes 原始图像字节流，支持可被 stb 解码的常见格式。
     * @return 返回二元组，依次为预测类别 ID 与对应置信度。
     */
    std::pair<int, float> Predict(const std::vector<uint8_t>& image_bytes);

    /**
     * @brief 对一批图像执行批量推理。
     *
     * @param batch_image_bytes 一组原始图像字节流，顺序与返回结果一一对应。
     * @return 返回与输入批次顺序一致的分类结果列表。
     */
    std::vector<std::pair<int, float>> PredictBatch(const std::vector<std::vector<uint8_t>>& batch_image_bytes);

private:
    /**
     * @brief 将原始图像字节流转换为模型输入张量。
     *
     * 该步骤负责图像解码、缩放、归一化以及内存布局转换，最终输出符合
     * ResNet18 输入要求的单样本浮点张量数据。
     *
     * @param image_bytes 原始图像字节流。
     * @return 适合直接构造输入张量的一维浮点数组。
     */
    std::vector<float> Preprocess(const std::vector<uint8_t>& image_bytes);

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    std::string input_name_ = "input";
    std::string output_name_ = "output";
};
