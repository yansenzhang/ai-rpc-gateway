#include <iostream>
#include <fstream>
#include <vector>
#include "inference_engine.h"

std::vector<uint8_t> ReadFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (file.read((char*)buffer.data(), size)) {
        return buffer;
    }
    throw std::runtime_error("Error reading file: " + filepath);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    try {
        std::cout << "Initializing Engine..." << std::endl;
        InferenceEngine engine(model_path);

        std::cout << "Reading Image: " << image_path << std::endl;
        std::vector<uint8_t> image_bytes = ReadFile(image_path);

        std::cout << "Running Prediction..." << std::endl;
        auto result = engine.Predict(image_bytes);

        std::cout << "Prediction Result:" << std::endl;
        std::cout << "Class ID: " << result.first << std::endl;
        std::cout << "Confidence: " << result.second << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}