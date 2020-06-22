# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-05-09 21:48:46
# Description: 推理流程

import json
import sys
from collections import OrderedDict

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from default import cfg
from faster_rcnn.data import COCODataset, Collater, DataSampler, build_transforms
from faster_rcnn.utils import last_checkpoint
from model import Model

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
cpu_device = torch.device('cpu')


def val(model):
    is_train = False
    global_step = 1

    # 加载数据集
    dataset = COCODataset(
        cfg,
        transforms=build_transforms(cfg),
        is_train=is_train
    )
    collater = Collater(cfg)
    sampler = DataSampler(dataset, cfg, global_step, is_train)
    dataloader = DataLoader(
        dataset,
        collate_fn=collater,
        batch_sampler=sampler,
        num_workers=2)

    model.eval()

    # 推理并保存结果
    predictions = {}
    for data in tqdm(dataloader):
        with torch.no_grad():
            images = data['images'].to(device)
            sizes = data['sizes'].to(device)
            ratios = data['ratios']
            img_ids = data['img_ids']

            bboxes, probs, labels = model(images, sizes)
            for img_id, ratio, bbox, prob, label in zip(img_ids, ratios, bboxes, probs, labels):
                predictions.update({img_id: [ratio, bbox.to(cpu_device), prob.to(cpu_device), label.to(cpu_device)]})

    image_ids = list(sorted(predictions.keys()))
    predictions = [predictions[i] for i in image_ids]

    # 将类别索引映射回原始状态
    map_classes = {v: int(k) for k, v in dataset.classes.items()}

    coco_det = []
    for image_id, (ratio, bbox, prob, label) in zip(image_ids, predictions):
        if len(bbox) == 0:
            continue
        bbox[:, 0::2] /= ratio[0]
        bbox[:, 1::2] /= ratio[1]
        x_min, y_min, x_max, y_max = bbox.split(1, dim=-1)
        # xyxy -> xywh
        bbox = torch.cat((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1), dim=-1)

        bbox = bbox.tolist()
        prob = prob.tolist()
        label = label.tolist()

        label = [map_classes[i] for i in label]

        coco_det.extend(
            [
                {
                    'image_id': image_id,
                    'category_id': label[i],
                    'bbox': box,
                    'score': prob[i],
                }
                for i, box in enumerate(bbox)
            ]
        )

    results = COCOResults('bbox')
    temp_name = 'det_tmp.json'
    with open(temp_name, 'w') as f:
        json.dump(coco_det, f)

    coco_dt = dataset.coco.loadRes(str(temp_name)) if results else COCO()
    coco_eval = COCOeval(dataset.coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results.update(coco_eval)
    print(results)


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results


if __name__ == '__main__':
    model = Model(cfg, is_train=False).to(device)

    checkpoint = last_checkpoint(cfg)
    if checkpoint:
        print('loading {}'.format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('weight not found')
        sys.exit()

    val(model)
