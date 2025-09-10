from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Graph, Tensor, Value
# from ultralytics.utils.checks import check_version

def make_anchors(feats: Tensor,
                 strides: Tensor,
                 grid_cell_offset: float = 0.5) -> Tuple[Tensor, Tensor]:
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device,
                          dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device,
                          dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class TRT_NMS(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: Graph,
            boxes: Tensor,
            scores: Tensor,
            iou_threshold: float = 0.65,
            score_threshold: float = 0.25,
            max_output_boxes: int = 100,
            background_class: int = -1,
            box_coding: int = 0,
            plugin_version: str = '1',
            score_activation: int = 0
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0,
                                 max_output_boxes, (batch_size, 1),
                                 dtype=torch.int32)
        boxes = torch.randn(batch_size, max_output_boxes, 4)
        scores = torch.randn(batch_size, max_output_boxes)
        labels = torch.randint(0,
                               num_classes, (batch_size, max_output_boxes),
                               dtype=torch.int32)

        return num_dets, boxes, scores, labels

    @staticmethod
    def symbolic(
            g,
            boxes: Value,
            scores: Value,
            iou_threshold: float = 0.45,
            score_threshold: float = 0.25,
            max_output_boxes: int = 100,
            background_class: int = -1,
            box_coding: int = 0,
            score_activation: int = 0,
            plugin_version: str = '1') -> Tuple[Value, Value, Value, Value]:
        out = g.op('TRT::EfficientNMS_TRT',
                   boxes,
                   scores,
                   iou_threshold_f=iou_threshold,
                   score_threshold_f=score_threshold,
                   max_output_boxes_i=max_output_boxes,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   outputs=4)
        nums_dets, boxes, scores, classes = out
        return nums_dets, boxes, scores, classes


class EfficientIdxNMS_TRT(torch.autograd.Function):
    """NMS with Index block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        iou_threshold: float = 0.65,
        score_threshold: float = 0.25,
        max_output_boxes: float = 100,
        box_coding: int = 1,
        background_class: int = -1,
        score_activation: int = 0,
        class_agnostic: int = 1,
        plugin_version: str = '1',
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
        det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        det_indices = torch.randint(0, num_boxes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        iou_threshold: float = 0.65,
        score_threshold: float = 0.25,
        max_output_boxes: float = 100,
        box_coding: int = 1,
        background_class: int = -1,
        score_activation: int = 0,
        class_agnostic: int = 1,
        plugin_version: str = '1',
    ) -> Tuple[Value, Value, Value, Value, Value]:
        out =  g.op(
            'TRT::EfficientIdxNMS_TRT',
            boxes,
            scores,
            outputs=5,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            score_threshold_f=score_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=background_class,
            score_activation_i=score_activation,
            class_agnostic_i=class_agnostic,
            plugin_version_s=plugin_version,
        )
        nums_dets, boxes, scores, classes, indices = out
        return nums_dets, boxes, scores, classes, indices
    
class C2f(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, 1))


class PostDetect(nn.Module):
    export = True
    shape = None
    dynamic = False
    iou_thres = 0.65
    conf_thres = 0.25
    topk = 100

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        shape = x[0].shape
        b, res, b_reg_num = shape[0], [], self.reg_max * 4
        for i in range(self.nl):
            res.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides

        return TRT_NMS.apply(boxes.transpose(1, 2), scores.transpose(1, 2),
                             self.iou_thres, self.conf_thres, self.topk)


class PostSeg(nn.Module):
    export = True
    shape = None
    dynamic = False

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size
        mc = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            2)  # mask coefficients
        boxes, scores, labels = self.forward_det(x)
        out = torch.cat([boxes, scores, labels.float(), mc.transpose(1, 2)], 2)
        return out, p.flatten(2)

    def forward_det(self, x):
        shape = x[0].shape
        b, res, b_reg_num = shape[0], [], self.reg_max * 4
        for i in range(self.nl):
            res.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = \
                (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides

        scores, labels = scores.transpose(1, 2).max(dim=-1, keepdim=True)
        return boxes.transpose(1, 2), scores, labels

class PostSegV2(nn.Module):
    export = True
    shape = None
    dynamic = False
    iou_thres = 0.65
    conf_thres = 0.2
    topk = 100

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        # bs, _, mask_h, mask_w = p.shape
        bs = p.shape[0]  # batch size
        mc = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            2)  # mask coefficients
        

        num_dets, boxes, scores, labels, indices = self.forward_det(x)
        # batch_indices = (
        #     torch.arange(bs, device=labels.device, dtype=labels.dtype).unsqueeze(1).expand(-1, self.topk).reshape(-1)
        # )
        # det_indices = indices.view(-1)
        # selected_masks = mc.transpose(1, 2)[batch_indices, det_indices].view(bs, self.topk, self.nm)
        # masks_protos = p.view(bs, self.nm, mask_h * mask_w)
        # det_masks = torch.matmul(selected_masks, masks_protos).sigmoid().view(bs, self.max_det, mask_h, mask_w)
        return  num_dets, boxes, scores, labels, indices, mc.transpose(1, 2), p.flatten(2)


    def forward_det(self, x):
        shape = x[0].shape
        b, res, b_reg_num = shape[0], [], self.reg_max * 4
        for i in range(self.nl):
            res.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides

        # return boxes, scores
        return EfficientIdxNMS_TRT.apply(boxes.transpose(1, 2), scores.transpose(1, 2),
                             self.iou_thres, self.conf_thres, self.topk)


class PostSegV3(nn.Module):
    export = True
    shape = None
    dynamic = False
    iou_thres = 0.65
    conf_thres = 0.2
    topk = 100

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs, _, mask_h, mask_w = p.shape
        bs = p.shape[0]  # batch size
        mc = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            2)  # mask coefficients
        

        num_dets, boxes, scores, labels, indices = self.forward_det(x)
        batch_indices = (
            torch.arange(bs, device=labels.device, dtype=labels.dtype).unsqueeze(1).expand(-1, self.topk).reshape(-1)
        )
        det_indices = indices.view(-1)
        selected_masks = mc.transpose(1, 2)[batch_indices, det_indices].view(bs, self.topk, self.nm) #1 100 32
        masks_protos = p.view(bs, self.nm, mask_h * mask_w) # 1 32 160*160
        det_masks = torch.matmul(selected_masks, masks_protos).sigmoid().view(bs, self.topk, mask_h, mask_w) #1 100 160 160
        # return  num_dets, boxes, scores, labels, indices, det_masks
        return  num_dets, boxes, scores, labels, det_masks


    def forward_det(self, x):
        shape = x[0].shape
        b, res, b_reg_num = shape[0], [], self.reg_max * 4
        for i in range(self.nl):
            res.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides
        return EfficientIdxNMS_TRT.apply(boxes.transpose(1, 2), scores.transpose(1, 2),
                             self.iou_thres, self.conf_thres, self.topk)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:

        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class UltralyticsSegment(nn.Module):
    """Ultralytics Segment head for segmentation models."""

    max_det = 100
    iou_thres = 0.45
    conf_thres = 0.25
    dynamic=False
    end2end = False  # end2end
    xyxy = False  # xyxy or xywh output

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs, _, mask_h, mask_w = p.shape
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2).permute(0, 2, 1)  # mask coefficients

        # Detect forward
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        dbox, cls = self._inference(x)

        ## Using transpose for compatibility with EfficientIdxNMS_TRT
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

        # Retrieve the corresponding masks using batch and detection indices.
        batch_indices = (
            torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1).expand(-1, self.max_det).reshape(-1)
        )
        det_indices = det_indices.view(-1)
        selected_masks = mc[batch_indices, det_indices].view(bs, self.max_det, self.nm)

        masks_protos = p.view(bs, self.nm, mask_h * mask_w)
        det_masks = torch.matmul(selected_masks, masks_protos).view(bs, self.max_det, mask_h * mask_w)
        # det_masks = torch.matmul(selected_masks, masks_protos).sigmoid().view(bs, self.max_det, mask_h, mask_w)

        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            det_masks
            #F.interpolate(det_masks, size=(mask_h * 4, mask_w * 4), mode="bilinear", align_corners=False).gt_(0.5).to(torch.int32) #uint8在trt8.5中不支持
        )

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and not (self.end2end or self.xyxy), dim=1)
    
    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return dbox, cls.sigmoid()
    
def optim(module: nn.Module, segv2=False, segv3=False):
    s = str(type(module))[6:-2].split('.')[-1]
    if s == 'Detect':
        setattr(module, '__class__', PostDetect)
    elif s == 'Segment' and segv2:
        setattr(module, '__class__', PostSegV2)
    elif s == 'Segment' and segv3:
        # raise ValueError("not support now!")
        setattr(module, '__class__', UltralyticsSegment)
    elif s == 'Segment':
        setattr(module, '__class__', PostSeg)
    elif s == 'C2f':
        setattr(module, '__class__', C2f)
