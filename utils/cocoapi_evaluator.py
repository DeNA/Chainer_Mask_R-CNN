import copy
import numpy as np

from chainer import reporter
import chainer.training.extensions

from utils import eval_detection_coco
from chainercv.utils import apply_prediction_to_iterator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class COCOAPIEvaluator(chainer.training.extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, iterator, target, ids, cocoanns, label_names=None):
        super(COCOAPIEvaluator, self).__init__(
            iterator, target)
        self.ids = ids
        self.cocoanns = cocoanns

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        annType = ['segm','bbox','keypoints']
        annType = annType[1]
        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_prediction_to_iterator(
            target.predict, it)
        # delete unused iterators explicitly
        del in_values

        pred_bboxes, _, pred_labels, pred_scores, pred_masks = out_values

        if len(rest_values) == 3:
            gt_bboxes, gt_labels, gt_difficults = rest_values
        elif len(rest_values) == 2:
            gt_bboxes, gt_labels = rest_values
            gt_difficults = None
        elif len(rest_values) == 5:
            gt_bboxes, gt_labels, _, _, i = rest_values
            gt_difficults = None
        pred_bboxes = iter(list(pred_bboxes))
        pred_labels = iter(list(pred_labels))
        pred_scores = iter(list(pred_scores))
        gt_bboxes = iter(list(gt_bboxes))
        gt_labels = iter(list(gt_labels))
        data_dict = []
        for i, (pred_bbox, pred_label, pred_score) in \
            enumerate(zip(pred_bboxes, pred_labels, pred_scores)):
            for bbox, label, score in zip(pred_bbox, pred_label, pred_score):
                A={"image_id":int(self.ids[i]), "category_id":int(label), "bbox":bbox.tolist(), "score":float(score)}
                data_dict.append(A)
        if len(data_dict)>0:
            cocoGt=self.cocoanns
            cocoDt=cocoGt.loadRes(data_dict)
            cocoEval = COCOeval(self.cocoanns, cocoDt, annType)
            cocoEval.params.imgIds  = [int(id_) for id_ in self.ids]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            report = {'map': cocoEval.stats[0]} # report COCO AP (IoU=0.5:0:95)
        else:
            report = {'map': 0}
        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation