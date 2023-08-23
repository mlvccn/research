import torch

def pytorch_iou(pred, target, obj_num, epsilon=1e-6):
    '''
    pred: [bs, h, w]
    target: [bs, h, w]
    obj_num: [bs]
    '''
    bs = obj_num.size(0)
    all_iou = []
    for idx in range(bs):
        now_pred = pred[idx].unsqueeze(0)
        now_target = target[idx].unsqueeze(0)
        now_obj_num = obj_num[idx]

        obj_ids = torch.arange(0, now_obj_num + 1, device=now_pred.device).int().view(-1, 1, 1)
        if obj_ids.size(0) == 1:  # only contain background
            continue
        else:
            obj_ids = obj_ids[1:]
            now_pred = (now_pred == obj_ids).float()
            now_target = (now_target == obj_ids).float()

            intersection = (now_pred * now_target).sum((1, 2))
            union = ((now_pred + now_target) > 0).float().sum((1, 2))

            now_iou = (intersection + epsilon) / (union + epsilon)

            all_iou.append(now_iou.mean())
    if len(all_iou) > 0:
        all_iou = torch.stack(all_iou).mean()
    else:
        all_iou = torch.ones((1), device=pred.device)
    return all_iou


    # gt = np.hstack(gt_for_eval).flatten()
    # p = np.hstack(pred_for_eval).flatten()
    # precision, recall, _ = precision_recall_curve(gt, p)
    # Fmax = 2 * (precision * recall) / (precision + recall)
    # mae = np.mean(np.abs(p - gt))
    # logging.info('Finished Inference F measure: {:.5f} MAE: {: 5f} IOU: {:5f}'
    #              .format(np.max(Fmax), mae, ious.avg))