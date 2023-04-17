**ONNX部署**：

onnxruntime 部署

```
auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());
```

报错：

```
2023-01-04 08:39:32.546550876 [E:onnxruntime:, sequential_executor.cc:369 Execute] Non-zero status code returned while running ReduceMax node. Name:'ReduceMax_1728' Status Message: /onnxruntime_src/onnxruntime/core/providers/cpu/reduction/reduction_ops.cc:763 void onnxruntime::ValidateKeepDims(const onnxruntime::TensorShape&, int64_t) keepdims was false. Can't reduce on dim with value of 0 if 'keepdims' is false. Invalid output shape would be produced. input_shape:{0,4}

terminate called after throwing an instance of 'Ort::Exception'
  what():  Non-zero status code returned while running ReduceMax node. Name:'ReduceMax_1728' Status Message: /onnxruntime_src/onnxruntime/core/providers/cpu/reduction/reduction_ops.cc:763 void onnxruntime::ValidateKeepDims(const onnxruntime::TensorShape&, int64_t) keepdims was false. Can't reduce on dim with value of 0 if 'keepdims' is false. Invalid output shape would be produced. input_shape:{0,4}
```

