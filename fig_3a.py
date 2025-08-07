import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.models.task_vectors import NonLinearTaskVector


# Functions for metrics
def nai(datasets, zsh, ft, merged):
    return {ds: (merged[ds] - zsh[ds]) / (ft[ds] - zsh[ds]) for ds in datasets}

def calc_rank(S, norm_thresh=0.95):
    # Rank based on approximation error (Eq. 6) in the paper
    rank = np.argmax(np.sqrt(np.cumsum(S.pow(2) / S.pow(2).sum())) > norm_thresh)
    return rank

def alignment_ratio(S, S_proj):
    # Subspace alignment ratio based on norms of projected task matrix vs norm of the original one (Eq. 5) in the paper
    return np.linalg.norm(S_proj, ord=2) / np.linalg.norm(S, ord=2)

def load_merging_results(filename, datasets):
    with open(filename, 'r') as file:
        data = json.load(file)
    return {f'{ds}': data['test'][f'{ds}:top1'] for ds in datasets}

def load_finetuned_results(filename, datasets):
    with open(filename, 'r') as file:
        data = json.load(file)
    return {f'{ds}': data[f'{ds}'] for ds in datasets}


MODELS_ROOT = os.path.expanduser('~/models/iso-ckpts')
datasets = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
model = 'ViT-B-16'
device = "cuda:0"

zeroshot_accs = {'Cars': 0.6460639223977117, 'DTD': 0.5212765957446809, 'EuroSAT': 0.5414814814814815, 'GTSRB': 0.4334125098970705, 'MNIST': 0.5179, 'RESISC45': 0.6576190476190477, 'SUN397': 0.6550125944584383, 'SVHN': 0.5198217578365089}
sum_accs_at_optimum = {'Cars': 0.6930730008705385, 'DTD': 0.5946808510638298, 'EuroSAT': 0.8492592592592593, 'GTSRB': 0.7468725257323832, 'MNIST': 0.9828, 'RESISC45': 0.7503174603174603, 'SUN397': 0.6525944584382871, 'SVHN': 0.8393131530424094}
iso_accs_at_optimum = {'Cars': 0.8343489615719438, 'DTD': 0.9175531914893617, 'EuroSAT': 0.9762962962962963, 'GTSRB': 0.9535233570863024, 'MNIST': 0.9911, 'RESISC45': 0.9225396825396825, 'SUN397': 0.7402015113350126, 'SVHN': 0.9097264904732637}
ft_accs = {'Cars': 0.8761348091033454, 'DTD': 0.9829787234042553, 'EuroSAT': 0.9888888888888889, 'GTSRB': 0.9900237529691212, 'MNIST': 0.9975, 'RESISC45': 0.9693650793650793, 'SUN397': 0.788110831234257, 'SVHN': 0.9782191149354641}

# Load results from logs if you have them
# zeroshot_accs = load_merging_results("logs/...", datasets)
# sum_accs_at_optimum = load_merging_results("logs/...", datasets)
# iso_accs_at_optimum = load_merging_results("logs/...", datasets)
# ft_accs = load_finetuned_results("results/single_task/ViT-B-16/nonlinear_ft_accuracies.json", datasets)

# Calculate NAI
nai_sum = nai(datasets, zeroshot_accs, ft_accs, sum_accs_at_optimum)
nai_iso = nai(datasets, zeroshot_accs, ft_accs, iso_accs_at_optimum)

# Calculate task vectors
task_vectors = []
pretrained_checkpoint = f'{MODELS_ROOT}/{model}/MNISTVal/nonlinear_zeroshot.pt'
for dataset in datasets:
    finetuned_checkpoint = f'{MODELS_ROOT}/{model}/{dataset}Val/nonlinear_finetuned.pt'
    task_vectors.append(NonLinearTaskVector(model, pretrained_checkpoint, finetuned_checkpoint))

# Calculate alignment
rank_threshold = 0.95
allignment_ratios_sum = {ds: [] for ds in datasets}
allignment_ratios_iso = {ds: [] for ds in datasets}
with torch.no_grad():
    for key in task_vectors[0].vector:
        _tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]

        if (len(_tvs[0].shape) == 2 and key.startswith("model.visual")):
            print(f"Processing {key}")
            
            merge_by_sum = sum(_tvs)
            U, S, V = torch.linalg.svd(merge_by_sum, full_matrices=False)
            sum_rel_rank = calc_rank(S.cpu(), norm_thresh=rank_threshold)
            U_sum_k = U[:, :sum_rel_rank]

            S_iso = torch.ones_like(S) * S.mean()
            iso_rel_rank = calc_rank(S_iso.cpu(), norm_thresh=rank_threshold)
            U_iso_k = U[:, :iso_rel_rank]

            for i, tv in enumerate(_tvs):
                U_tv, S_tv, V_tv = torch.linalg.svd(tv, full_matrices=False)

                proj_tv_onto_sum_k = torch.linalg.multi_dot((U_sum_k, U_sum_k.T, tv))
                U_tv_k, S_tv_k, V_tv_k = torch.linalg.svd(proj_tv_onto_sum_k, full_matrices=False)
                ar_sum = alignment_ratio(S_tv.cpu(), S_tv_k.cpu())
                allignment_ratios_sum[datasets[i]].append(ar_sum)
                print(f"Alignment ratio for {datasets[i]}->SUM_k: {ar_sum:.4f}")

                proj_tv_onto_iso_k = torch.linalg.multi_dot((U_iso_k, U_iso_k.T, tv))
                U_iso_k, S_iso_k, V_iso_k = torch.linalg.svd(proj_tv_onto_iso_k, full_matrices=False)
                ar_iso = alignment_ratio(S_tv.cpu(), S_iso_k.cpu())
                allignment_ratios_iso[datasets[i]].append(ar_iso)
                print(f"Alignment ratio for {datasets[i]}->ISO_k: {ar_iso:.4f}")
        else:
            print(f"Skipping {key}")

avg_alignment_ratios_sum = {ds: np.mean(ar) for ds, ar in allignment_ratios_sum.items() if len(ar) > 0}
print(f"Average alignment ratios for sum: {avg_alignment_ratios_sum}")
avg_alignment_ratios_iso = {ds: np.mean(ar) for ds, ar in allignment_ratios_iso.items() if len(ar) > 0}
print(f"Average alignment ratios for iso: {avg_alignment_ratios_iso}")

# Plot
fig, axes = plt.subplots(1, 1, figsize=(6, 5))
for ds in datasets:
    lines = plt.plot(avg_alignment_ratios_sum[ds], nai_sum[ds], 
                     marker='*', markersize=16, linestyle='None', alpha=0.9,
                     label=f'TA:{ds}')
    plt.plot(avg_alignment_ratios_iso[ds], nai_iso[ds], 
             marker='^', markersize=16, linestyle='None', alpha=0.9,
             label=f'Iso:{ds}', 
             color=lines[-1].get_color())

plt.xlabel("SAR avg")
plt.ylabel("NAI")
plt.legend(ncol=2,loc='lower right')
plt.grid(True)
plt.tight_layout()

fig_path = f"fig/ta-iso-nai-vs-ar"
fig.savefig(f"{fig_path}.png")
