import torch
import torchvision.models as models

def main():
    print("Loading pretrained ResNet18 model...")
    # 加载预训练的 ResNet18 模型权重
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    # 创建 Dummy Input，设置 batch_size 为 1，3通道，224x224 尺寸
    dummy_input = torch.randn(1, 3, 224, 224)

    onnx_file_path = "resnet18.onnx"
    print(f"Exporting model to {onnx_file_path} with dynamic batch size...")

    # 导出模型为 ONNX 格式
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},   # 输入的第一维是动态的 batch_size
            'output': {0: 'batch_size'}   # 输出的第一维是动态的 batch_size
        }
    )
    print("Export completed successfully.")

if __name__ == "__main__":
    main()
