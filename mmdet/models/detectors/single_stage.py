import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

import os.path as osp
from mmcv.image import tensor2imgs


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_points=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        label_img, unlabel_img = img[:len(img) // 2], img[len(img) // 2:]
        if '_' not in img_metas[3]['ori_filename'] or '_' not in img_metas[4]['ori_filename'] or '_' not in img_metas[5]['ori_filename']:
            print(img_metas[3]['ori_filename'])
            print('warnnings, there is a error, we need a _ in the ori_filename, but there has not')

        for i in range(len(gt_points)):
            gt_points[i] = gt_points[i][..., :2]
        
        label_img_metas, unlabel_img_metas = img_metas[:len(img_metas) // 2], img_metas[len(img_metas) // 2:]
        label_gt_bboxes, unlabel_gt_bboxes = gt_bboxes[:len(gt_bboxes) // 2], gt_bboxes[len(gt_bboxes) // 2:]
        label_gt_labels, unlabel_gt_labels = gt_labels[:len(gt_labels) // 2], gt_labels[len(gt_labels) // 2:]
        label_gt_points, unlabel_gt_points = gt_points[:len(gt_points) // 2], gt_points[len(gt_points) // 2:]
        label_type2weight = self.train_cfg.label_type2weight
        # for index in range(len(gt_points)):
        #      gt_points[index] = torch.cat([gt_points[index], gt_points[index]], dim=-1)
        #      gt_points[index][..., 2:] = gt_points[index][..., 2:] + 1
        # gt_points = torch.cat([gt_points, gt_points], dim=-1)
        # gt_points[..., 2:] = gt_points[..., 2:] + 1
        
        # 可视化
        # for index in range(len(gt_points)):
        #     gt_points[index] = torch.cat([gt_points[index], torch.ones([gt_points[index].shape[0],1]).to(gt_points[index].device)], dim=-1)
        #
        # bbox_results = [
        #     bbox2result(det_bboxes, torch.zeros(det_bboxes.shape[0]), self.bbox_head.num_classes)
        #     for det_bboxes, _ in zip(gt_points, gt_labels)
        # ]

        # imgs = tensor2imgs(img, **img_metas[0]['img_norm_cfg'])
        # out_dir = './work_dirs/fcos_r50_caffe_fpn_gn-head_1x_coco/out_labeled/'
        # for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        #     h, w, _ = img_meta['img_shape']
        #     img_show = img[:h, :w, :]
        #     if out_dir:
        #         out_file = osp.join(out_dir, img_meta['ori_filename'])
        #     else:
        #         out_file = None
        #     self.show_result(
        #         img_show,
        #         bbox_results[i],
        #         show=True,
        #         out_file=out_file,
        #         score_thr=0.3)

        # for index in range(len(unlabel_gt_bboxes)):
        #     unlabel_gt_bboxes[index] = torch.cat([unlabel_gt_bboxes[index], torch.ones([unlabel_gt_bboxes[index].shape[0],1]).to(unlabel_gt_bboxes[index].device)], dim=-1)
        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #     for det_bboxes, det_labels in zip(unlabel_gt_bboxes, unlabel_gt_labels)
        # ]

        # imgs = tensor2imgs(unlabel_img, **unlabel_img_metas[0]['img_norm_cfg'])
        # out_dir = './work_dirs/fcos_r50_caffe_fpn_gn-head_1x_coco/out_unlabeled/'
        # for i, (img, img_meta) in enumerate(zip(imgs, unlabel_img_metas)):
        #     h, w, _ = img_meta['img_shape']
        #     img_show = img[:h, :w, :]
        #     if out_dir:
        #         out_file = osp.join(out_dir, img_meta['ori_filename'])
        #     else:
        #         out_file = None
        #     self.show_result(
        #         img_show,
        #         bbox_results[i],
        #         show=True,
        #         out_file=out_file,
        #         score_thr=0.3)

        x = self.extract_feat(label_img)
        label_losses = self.bbox_head.forward_train(x, label_img_metas, label_gt_bboxes,
                                              label_gt_labels, label_gt_points, gt_bboxes_ignore, supervised_type='box')

        x = self.extract_feat(unlabel_img)
        unlabel_losses = self.bbox_head.forward_train(x, unlabel_img_metas, unlabel_gt_bboxes,
                                              unlabel_gt_labels, unlabel_gt_points, gt_bboxes_ignore, supervised_type='point')

        for k in label_losses.keys():
            label_losses[k] += unlabel_losses[k] * label_type2weight[1]
        return label_losses

        # x = self.extract_feat(img)
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore)
        # return losses
        # return label_losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels
