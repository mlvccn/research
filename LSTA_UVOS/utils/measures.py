from collections import OrderedDict as odict
import numpy as np
from skimage.morphology import binary_dilation, disk


def evaluate_sequence(segmentations, annotations, object_info, measure='J'):
    """
    Evaluate video sequence results.

      Arguments:
          segmentations (dict of ndarray): segmentation labels.
          annotations   (dict of ndarray): ground-truth labels.
          object_info   dict: {object_id: first_frame_index}

      measure       evaluation metric (J,F)
    """

    results = dict(raw=odict())

    _measures = {'J': davis_jaccard_measure, 'F': davis_f_measure}
    _statistics = {'decay': decay, 'mean': mean, 'recall': recall, 'std': std}

    for obj_id, first_frame in object_info.items():

        r = np.ones((len(annotations))) * np.nan

        for i, (an, sg) in enumerate(zip(annotations, segmentations)):
            if list(annotations.keys()).index(first_frame) < i < len(annotations) - 1:
                r[i] = _measures[measure](annotations[an].squeeze(0).numpy() == obj_id,
                                          segmentations[sg].squeeze(0).numpy() == obj_id)

        results['raw'][obj_id] = r

    for stat, stat_fn in _statistics.items():
        results[stat] = [float(stat_fn(r)) for r in results['raw'].values()]

    return results


# Originally db_eval_iou() in the davis challenge toolkit:
def davis_jaccard_measure(fg_mask, gt_mask):
    """ Compute region similarity as the Jaccard Index.

    :param fg_mask: (ndarray): binary segmentation map.
    :param gt_mask: (ndarray): binary annotation map.
    :return: jaccard (float): region similarity
    """

    gt_mask = gt_mask.astype(np.bool)
    fg_mask = fg_mask.astype(np.bool)

    if np.isclose(np.sum(gt_mask), 0) and np.isclose(np.sum(fg_mask), 0):
        return 1
    else:
        return np.sum((gt_mask & fg_mask)) / \
               np.sum((gt_mask | fg_mask), dtype=np.float32)


# Originally db_eval_boundary() in the davis challenge toolkit:
def davis_f_measure(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F

if __name__ == '__main__':
    import os
    import glob
    from PIL import Image
    import torch
    results_dir = "/home/zhangyu/projects/ETAN_A/logs/rn101_ytb_ohem_memory_kv_lr3e_3/eval/yto/yto_rn101_ytb_ohem_memory_kv_lr3e_3_ckpt_14000/Annotations/"
    # results_dir = "./logs/rn101_dv_ohem_memory_kv_lr3e_3_from_ytb/eval/yto/yto_rn101_dv_ohem_memory_kv_lr3e_3_from_ytb_ckpt_7500/Annotations"
    image_set_dir = './data/YTO/ImageSets/test.txt'
    gtbase_dir = './data/YTO/Annotations'
    with open(image_set_dir,'r') as f:
        seqnames = f.readlines()
        seq_names = [i.strip() for i in seqnames]
    
    j_score_dict = {}
    j_mean = []
    for name in seq_names:
        res_seq_dir = os.path.join(results_dir,name)
        gt_seq_dir = os.path.join(gtbase_dir,name)
        print(gt_seq_dir)
        gt_dir_list = sorted(glob.glob(gt_seq_dir+"/*.png"))
        res_dir_list = sorted(glob.glob(res_seq_dir+"/*.png"))
        assert len(gt_dir_list) == len(res_dir_list), "expect {} predicted mask {}, but got {}".format(name, len(gt_dir_list), len(res_dir_list))
        # assert len(gt_dir_list) == len(res_dir_list), gt_seq_dir
        # j_score_dict[name]
        seq_j_scores = []
        for i in range(len(gt_dir_list)):
            gt_dir = gt_dir_list[i]
            res_dir = res_dir_list[i]
            gt = Image.open(gt_dir)
            res = Image.open(res_dir)
            gt_arr = np.array(gt)
            res_arr = np.array(res)
            if gt_arr.shape!= res_arr.shape:
                # print("reshape")
                h,w = gt_arr.shape
                res_tensor = torch.from_numpy(res_arr).float().unsqueeze(0).unsqueeze(0)
                res_tensor_ = torch.nn.functional.interpolate(res_tensor, (h,w), mode='nearest')
                res_arr = res_tensor_.squeeze().numpy()
            j_score = davis_jaccard_measure(res_arr, gt_arr)
            seq_j_scores.append(j_score)
            print(j_score)
        j_score_dict[name] = np.mean(seq_j_scores)
        if len(seq_j_scores) < 2:
            continue
        j_mean.append(np.mean(seq_j_scores))
    
    print(j_score_dict)
    print(np.mean(j_mean))
