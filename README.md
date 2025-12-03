# <center> AutoOrgan 

AutoOrgan commercial version.

步骤：
1. 运行test_generate_iv生成一个iv文件
2. 运行pth2onnx.py把nnUnet的模型转换为onnx模型.
3. 运行test_OnnxCryptor 把onnx模型进行加密，
4. 运行test_package_files把两个配置文件和加密后的onnx模型和iv文件打包成一个pkg文件.