import os
from PIL import Image
import numpy  as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from metrics import calculate_precision_at_k_and_iou_metrics
import pycocotools.mask as mask_util


def load_saved_masks(mask_dir):
    mask_dir_list = sorted(os.listdir(mask_dir))
    predictions = []
    for mask_name in mask_dir_list:
        d = os.path.join(mask_dir, mask_name)
        img_name_list = mask_name[:-4].split("_")
        img_id = "_".join(img_name_list[:-1])
        # "/home/zhangyu/projects/ref-vos/MTTR/runs/2022_03_24-09_10_59_PM/validation_outputs/epoch_0/Annotations/v_zITqe1ong0Q_f_105_i_0_93974.png"
        msk = np.array(Image.open(d))
        msk = np.array(msk > 127, dtype=np.uint8)
        m = mask_util.encode(np.array(msk[ :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                
        predictions.append({'image_id': img_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': 1.0})
    return predictions


def main():
    d1 = "/home/zhangyu/projects/ref-vos/MTTR/runs/2022_03_24-09_40_05_PM/validation_outputs/epoch_139/Annotations"
    d2 = "/home/zhangyu/projects/ref-vos/MTTR/runs/2022_03_24-09_10_59_PM/validation_outputs/epoch_0/Annotations/"
    predictions = load_saved_masks(d1)
    predictions2 = load_saved_masks(d2)
    predictions.extend(predictions2)
    coco_gt = COCO('./datasets/a2d_sentences/a2d_sentences_test_annotations_in_coco_format.json')
    coco_pred = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')

    coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
    ap_metrics = coco_eval.stats[:6]
    eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}

    precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
    eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
    eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
    print(eval_metrics)

if __name__ == "__main__":
    main()