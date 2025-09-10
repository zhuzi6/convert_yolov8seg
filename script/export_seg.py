import argparse
import onnx
import torch
import onnxsim

from io import BytesIO
from ultralytics import YOLO
from common import optim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='PyTorch yolov8 weights')
    parser.add_argument('--opset',
                        type=int,
                        default=11,
                        help='ONNX opset version')
    parser.add_argument('--sim',
                        action='store_true',
                        help='simplify onnx model')
    parser.add_argument('--topk',
                        type=int,
                        default=100,
                        help='Max number of detection bboxes')
    parser.add_argument('--segv2',
                        action='store_true',
                        help='use efficient nms')
    parser.add_argument('--segv3',
                        action='store_true',
                        help='use efficient nms')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=[1, 3, 640, 640],
                        help='Model input shape only for api builder')
    parser.add_argument('--save_path',
                        type=str,
                        default='',
                        help='Export ONNX device')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Export ONNX device')
    
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):
    b = args.input_shape[0]
    YOLOv8 = YOLO(args.weights)
    model = YOLOv8.model.fuse().eval()
    for m in model.modules():
        optim(m, segv2=args.segv2, segv3=args.segv3)
        m.to(args.device)
    model.to(args.device)
    fake_input = torch.randn(args.input_shape).to(args.device)

    if args.save_path == '':
        save_path = args.weights.replace('.pt', '.onnx')
    else:
        save_path = args.save_path

    for _ in range(2):
        model(fake_input)
    if args.segv2:
        output_names = ['num_dets', 'bboxes', 'scores', 'labels', 'indices', 'mask_coefficients', 'mask_protos']
    elif args.segv3:
        output_names = ["num_dets", "det_boxes", "det_scores", "det_classes", "masks"]
    else:
        output_names = ['outputs', 'proto']

    with BytesIO() as f:
        torch.onnx.export(model,
                          fake_input,
                          f,
                          opset_version=args.opset,
                          input_names=['images'],
                          output_names=output_names)
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    if args.segv2:
        shapes = [b, 1, b, args.topk, 4, b, args.topk, b, args.topk, b, args.topk]
        for i in onnx_model.graph.output[:5]:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))
    elif args.segv3:
        B, C, H, W = args.input_shape
        shapes = [b, 1, b, args.topk, 4, b, args.topk, b, args.topk, b, args.topk, int(H//4)*int(W//4)]
        for i in onnx_model.graph.output[:5]:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

    if args.sim:
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'assert check failed'
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')


if __name__ == '__main__':
    main(parse_args())

