import torch
import torch.multiprocessing as mp
import timeit

def get_optimal_thresholds(pred: torch.tensor, label: torch.tensor, num_classes: int):
    device = pred.device
    batch_size = pred.shape[0]
    thresholds = torch.full((batch_size, num_classes), 0.5, device=device)

    def getIoU(pred: torch.tensor, label: torch.tensor) -> torch.tensor:
        intersection = torch.zeros(batch_size, num_classes, device=device)
        union = torch.zeros(batch_size, num_classes, device=device)
        
        for cls in range(num_classes):
            pred_mask = (pred == cls)
            true_mask = (label == cls)
            intersection[:, cls] = torch.logical_and(pred_mask, true_mask).sum(dim=(1,2)).float()
            union[:, cls] = torch.logical_or(pred_mask, true_mask).sum(dim=(1,2)).float()
        
        iou = intersection / (union + 1e-10)  # adding epsilon to avoid division by zero
        return iou

    def get_H_score(thresholds: torch.tensor) -> torch.tensor:
        pseudo_label = get_pseudo_mask(thresholds)
        iou = getIoU(pseudo_label, label)
        miou_known = torch.mean(iou[:, :-1], dim=1)
        H_score = (2 * miou_known * iou[:, -1]) / (miou_known + iou[:, -1])
        return H_score
    def get_pseudo_mask(thresholds: torch.tensor):
        pseudo_prob, pseudo_label = torch.max(pred, dim=1)
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).expand_as(pseudo_label)
        unknown_mask = pseudo_prob < thresholds[batch_indices, pseudo_label]
        pseudo_label[unknown_mask] = (num_classes - 1)
        return pseudo_label

    best_thresholds = thresholds.clone()
    best_H_scores = get_H_score(thresholds)

    # Parallel optimization for all classes and all batch items
    low = torch.zeros(batch_size, num_classes - 1, device=device)
    high = torch.ones(batch_size, num_classes - 1, device=device)
    cntr = 0
    while torch.max(high - low) > 5e-3:
        cntr += 1
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        H_scores1 = torch.zeros(batch_size, num_classes - 1, device=device)
        H_scores2 = torch.zeros(batch_size, num_classes - 1, device=device)

        for i in range(num_classes - 1):
            thresholds[:, i] = mid1[:, i]
            H_scores1[:, i] = get_H_score(thresholds)
            thresholds[:, i] = mid2[:, i]
            H_scores2[:, i] = get_H_score(thresholds)
            thresholds[:, i] = best_thresholds[:, i]  # Reset to best known threshold

        mask = H_scores1 > H_scores2
        high[mask] = mid2[mask]
        low[~mask] = mid1[~mask]
    thresholds[:, :-1] = (low + high) / 2
    final_H_scores = get_H_score(thresholds)
    final_pseudo_labels = get_pseudo_mask(thresholds)
    return final_pseudo_labels, final_H_scores.cpu(), thresholds.cpu()

# Example usage

torch.manual_seed(42)
batch_size = 8
A_512_512 = torch.rand(batch_size, 20, 512, 512)
B_512_512 = torch.randint(0, 20, (batch_size, 512, 512))
A_512_512 = A_512_512.cuda()
B_512_512 = B_512_512.cuda()

start_time = timeit.default_timer()
pseudo_label, h_scores, thresholds = get_optimal_thresholds_batch(A_512_512, B_512_512, 20)
print(f"Time: {timeit.default_timer() - start_time}")
print(f"Pseudo Label:{pseudo_label.shape}")
print(f"H-Scores: {h_scores}")
print(f"Thresholds: {thresholds.shape}")

'''
torch.manual_seed(42)
A = torch.rand(3, 3, 3)
A = torch.softmax(A, dim=0)
B = torch.tensor([[0, 2, 1],[2, 2, 2],[1, 1, 0]])
A_concat = torch.stack([A, A.clone()], dim=0)
B_concat = torch.stack([B, B.clone()], dim=0)

A_concat = A_concat.cuda()
B_concat = B_concat.cuda()
start_time = timeit.default_timer()
h_scores, thresholds = get_optimal_thresholds_batch(A_concat, B_concat, 3)
print(f"Time: {timeit.default_timer() - start_time}")
print(f"H-Scores: {h_scores}")
print(f"Thresholds: {[dict(zip(range(3), t)) for t in thresholds]}")
'''