import torch
import torch.multiprocessing as mp
import timeit

def get_optimal_thresholds(pred: torch.tensor, label: torch.tensor, num_classes: int):
    device = pred.device
    thresholds = torch.full((num_classes,), 0.5, device=device)

    def getIoU(pred: torch.tensor, label: torch.tensor) -> torch.tensor:
        intersection = torch.zeros(num_classes, device=device)
        union = torch.zeros(num_classes, device=device)
        for cls in range(num_classes):
            pred_mask = (pred == cls)
            true_mask = (label == cls)
            intersection[cls] = torch.logical_and(pred_mask, true_mask).sum().float()
            union[cls] = torch.logical_or(pred_mask, true_mask).sum().float()
        iou = intersection / (union + 1e-10)
        return iou

    def get_H_score_fast(thresholds: torch.tensor) -> float:
        pseudo_prob, pseudo_label = torch.max(pred, dim=0)
        unknown_mask = pseudo_prob < thresholds[pseudo_label]
        pseudo_label[unknown_mask] = num_classes
        iou = getIoU(pseudo_label, label)
        miou_known = torch.mean(iou[:-1])
        H_score = (2 * miou_known * iou[-1]) / (miou_known + iou[-1])
        return H_score.item()

    best_thresholds = thresholds.clone()
    best_H_score = get_H_score_fast(thresholds)

    # Parallel optimization for all classes
    low = torch.zeros(num_classes - 1, device=device)
    high = torch.ones(num_classes - 1, device=device)

    while torch.max(high - low) > 1e-3:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        H_scores1 = torch.zeros(num_classes - 1, device=device)
        H_scores2 = torch.zeros(num_classes - 1, device=device)

        for i in range(num_classes - 1):
            thresholds[i] = mid1[i]
            H_scores1[i] = get_H_score_fast(thresholds)
            thresholds[i] = mid2[i]
            H_scores2[i] = get_H_score_fast(thresholds)
            thresholds[i] = best_thresholds[i]  # Reset to best known threshold

        mask = H_scores1 > H_scores2
        high[mask] = mid2[mask]
        low[~mask] = mid1[~mask]

    thresholds[:-1] = (low + high) / 2
    final_H_score = get_H_score_fast(thresholds)

    return final_H_score, thresholds.cpu().numpy()

'''
# Example usage
torch.manual_seed(42)
A_512_512 = torch.rand(20, 512, 512)
B_512_512 = torch.randint(0, 20, (512, 512))
A_512_512 = A_512_512.cuda()
B_512_512 = B_512_512.cuda()

start_time = timeit.default_timer()
h, thres = get_optimal_thresholds(A_512_512, B_512_512, 20)
print(f"Time: {timeit.default_timer() - start_time}")
print(f"H-Score: {h}")
print(f"Threshold: {thres}")
'''