# YOLOV8SEG TensorRT
用于支持YOLOV8SEG的Torch模型转ONNX，并将ONNX转为带efficientidxnms插件的模型

# 环境依赖
onnx  
torch  
onnx-simplifier  
ultralytics

# 转ONNX
```
det:
python script/export_det.py --weights weights/**.pt --iou-thres 0.65 --conf-thres 0.2 --topk 100 --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0
seg:
不指定--save_path会保存在torch_model_path的相同路径下，--sim用于onnx的图优化，--segv3
python script/export_seg.py -w torch_model_path --sim --segv3 --save_path(optimal)
转成功后显示：
ONNX export success, saved as **.onnx
```

# 转TensorRT
```
det:
cd tensorRT/bin
./trtexec --onnx=yolov8n-pose.onnx --saveEngine=yolov8n-pose.engine
seg:
在CMakeLists.txt中把tensorrt的路径修改为自己的.
**.onnx表示onnx的路径  **.engine表示转化后tensorrt的路径 fp16表示以float16的精度转tensorrt

mkdir build
cmake ..
make -j8
./export **.onnx **.engine fp16

转成功后显示：
using kFP16!Succeeded parsing .onnx file!
Succeeded building serialized engine!
Succeeded building engine!
Succeeded saving .plan file!
Binding all? Yes
Bind[0]:i[0]->FP32  (1, 3, 640, 640) images
Bind[1]:o[0]->INT32 (1, 1) num_dets
Bind[2]:o[1]->FP32  (1, 100, 4) det_boxes
Bind[3]:o[2]->FP32  (1, 100) det_scores
Bind[4]:o[3]->INT32 (1, 100) det_classes
Bind[5]:o[4]->FP32  (1, 100, 25600) masks
id : 0, 4915200
id : 1, 4
id : 2, 1600
id : 3, 400
id : 4, 400
id : 5, 10240000
```