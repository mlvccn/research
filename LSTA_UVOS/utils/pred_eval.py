from collections import OrderedDict as odict
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import os
from glob import glob
# from utils.measures import evaluate_sequence,nanmean,mean

import warnings
from collections import OrderedDict as odict

import numpy as np
from skimage.morphology import binary_dilation, disk
from math import floor


# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
# -----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

# Originally db_eval_sequence() in the davis challenge toolkit:
def evaluate_sequence(segmentations, annotations, measure='J'):
    """
    Evaluate video sequence results.

      Arguments:
          segmentations (dict of ndarray): segmentation labels.
          annotations   (dict of ndarray): ground-truth labels.
          object_info   dict: {object_id: first_frame_index}

      measure       evaluation metric (J,F)
    """

    results = {"measure":measure}

    _measures = {'J': davis_jaccard_measure, 'F': davis_f_measure}
    _statistics = {'decay': decay, 'mean': mean, 'recall': recall, 'std': std}


    r = np.ones((len(annotations))) * np.nan

    for i, (an, sg) in enumerate(zip(annotations, segmentations)):
        if 0 < i < len(annotations) - 1:
            r[i] = _measures[measure](annotations[an].squeeze(0).numpy() != 0,
                                        segmentations[sg].squeeze(0).numpy() != 0)

    for stat, stat_fn in _statistics.items():
        results[stat] = [float(stat_fn(r))]

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


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + floor((y - 1) + height / h)
                    i = 1 + floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def nanmean(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(*args, **kwargs)


def mean(X):
    """
    Compute average ignoring NaN values.
    """
    return np.nanmean(X)


def recall(X, threshold=0.5):
    """
    Fraction of values of X scoring higher than 'threshold'
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = X[~np.isnan(X)]
        x = mean(x > threshold)
    return x


def decay(X, n_bins=4):
    """
    Performance loss over time.
    """
    X = X[~np.isnan(X)]
    ids = np.round(np.linspace(1, len(X), n_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [X[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])
    return D


def std(X):
    """
    Compute standard deviation.
    """
    return np.nanstd(X)


def text_bargraph(values):
    blocks = np.array(('u', ' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', 'o'))
    nsteps = len(blocks) - 2 - 1
    hstep = 1 / (2 * nsteps)
    values = np.array(values)
    nans = np.isnan(values)
    values[nans] = 0  # '░'
    indices = ((values + hstep) * nsteps + 1).astype(np.int)
    indices[values < 0] = 0
    indices[values > 1] = len(blocks) - 1
    graph = blocks[indices]
    graph[nans] = '░'
    graph = str.join('', graph)
    return graph


def imread(filename):
    im = np.array(Image.open(filename))
    im = np.ascontiguousarray(np.atleast_3d(im).transpose(2, 0, 1))
    im = torch.from_numpy(im)
    return im


def evaluate_dataset(prediction_dir, annotation_dir, dataset_name="davsod", single=True, measures=['J'], split='test_easy'):
    seq_names_list = sorted(os.listdir(prediction_dir))
    # seq_names_list_ = sorted(os.listdir(annotation_dir))
    # assert len(seq_names_list) == len(seq_names_list_), "video sequence number should be same!"

    K_seq_name_V_abs_dir_pred = {}
    K_seq_name_V_abs_dir_anno = {}
    for seq_name in seq_names_list:
        frame_pred_abs_dir_list = sorted(glob(os.path.join(prediction_dir, seq_name) + "/*.png"))
        frame_anno_abs_dir_list = sorted(glob(os.path.join(annotation_dir, seq_name) + '/*.png'))
        assert len(frame_anno_abs_dir_list) == len(frame_pred_abs_dir_list), \
            "Video Seq {} should have {} prediction, but only got{}.".format(
                seq_name, 
                len(frame_anno_abs_dir_list), 
                len(frame_pred_abs_dir_list)
            )
        K_seq_name_V_abs_dir_pred[seq_name] = frame_pred_abs_dir_list
        K_seq_name_V_abs_dir_anno[seq_name] = frame_anno_abs_dir_list

    for measure in measures:
        results = odict()
        dset_scores = []
        dset_decay = []
        dset_recall = []

        f = open(("evaluation-%s.txt" % measure), "w")

        def _print(msg):
            print(msg)
            print(msg, file=f)
            f.flush()

        for seq_idx, seq_name in enumerate(seq_names_list):
            annotations = odict()
            segmentations = odict()
            for frame_idx, (an_dir, sg_dir) in enumerate(zip(K_seq_name_V_abs_dir_pred[seq_name], K_seq_name_V_abs_dir_anno[seq_name])):
                anno = imread(an_dir)
                image = imread(sg_dir)
                annotations[frame_idx] = (anno != 0).byte() if single else anno
                # import pdb; pdb.set_trace()
                segmentations[frame_idx] = F.interpolate(image.unsqueeze(0), anno.shape[-2:], mode='nearest').squeeze(0)

            results = evaluate_sequence(segmentations=segmentations, annotations=annotations, measure=measure)
            # print(seq_name,results)

            # Print scores, per frame and object, ignoring NaNs

            # Print mean object score per frame and final score

            dset_decay.extend(results['decay'])
            dset_recall.extend(results['recall'])
            dset_scores.extend(results['mean'])

            # Print sequence results

            _print("Seq{}: {} Mean={:.4f}; Recall={:.4f}; Decay={:.4f}".format(seq_name ,measure, results['mean'][0],
                results['recall'][0], results['decay'][0]))

        _print("%s: %.3f, recall: %.3f, decay: %.3f" % (measure, mean(dset_scores), mean(dset_recall),
                                                        mean(dset_decay)))

        f.close()

# prediction_dir = "/home/ping501f/project/ETAN_A/logs/rn101_davsod_ohem_memory_kv_lr3e_3_from_ytb_0401/eval/davsod/davsod_rn101_davsod_ohem_memory_kv_lr3e_3_from_ytb_0401_ckpt_2500/Annotations/"
prediction_dir = "/home/zhangyu/projects/ETAN_A/logs/rn101_dv_ohem_memory_kv_lr3e_3_from_ytb/eval/yto/yto_rn101_dv_ohem_memory_kv_lr3e_3_from_ytb_ckpt_7500/Annotations/"
anno_dir = "/home/zhangyu/projects/ETAN_A/data/YTO/Annotations/"
evaluate_dataset(prediction_dir, anno_dir)