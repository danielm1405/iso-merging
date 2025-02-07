import copy
import time

import numpy as np
import torch
import tqdm

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.models.heads import get_classification_head
from src.models.modeling import ImageClassifier
from src.models.task_vectors import _Checkpoint, _TaskVector
from src.utils import utils
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_confusion_matrix,
)


def eval_single_dataset(image_encoder, dataset_name, args):
    start_time = time.time()
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0

        for _, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        top1 = correct / n

    metrics = {"top1": top1}
    dt = time.time() - start_time
    print(
        f"Done evaluating on {dataset_name}.\t Accuracy: {100*top1:.2f}%.\t Total time: {dt:.2f}s"
    )

    return metrics


def evaluate(
    pretrained_checkpoint,
    task_vector,
    args,
    scaling_coef,
    eval_masks=None,
):
    per_dataset_results = {}

    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )

    if eval_masks != None:
        assert args.method.name in ["tall_mask", "mag_masking"]
    else:
        image_encoder = task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=scaling_coef, args=args
        )

    for dataset_name in eval_datasets:

        if eval_masks != None:
            sparse_task_vector = copy.deepcopy(task_vector)
            # remove "Val" from dataset_name
            mask = (
                eval_masks[dataset_name[:-3]]
                if "Val" in dataset_name
                else eval_masks[dataset_name]
            )
            # apply mask to sparsify the task vectors with Hadamard product
            sparse_task_vector.vector = {
                k: sparse_task_vector.vector[k] * mask[k].bool().cpu()
                for k in mask.keys()
            }
            # reconstruct theta_t^
            image_encoder = sparse_task_vector.apply_to(
                pretrained_checkpoint, scaling_coef=1.0, args=args
            )

        # evalute performance
        results = eval_single_dataset(image_encoder, dataset_name, args)
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    task_vector: _TaskVector,
    pretrained_checkpoint: _Checkpoint,
    args,
    scaling_coef: float,
    eval_masks=None,
):
    start_time = time.time()

    coef_info = evaluate(
        pretrained_checkpoint,
        task_vector,
        args,
        scaling_coef,
        eval_masks,
    )

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    print(f"Total evaluation time: {time.time() - start_time:.2f}s")
    return coef_info


def evaluate_task_vector(
    task_vector, pretrained_checkpoint, args, eval_masks=None
):
    info = {}

    if args.method.name == "tall_mask" or eval_masks is not None:
        scaling_coef_range = [1.0]
    elif args.method.name == "zeroshot":
        scaling_coef_range = [0.0]
    elif args.method.name == "average":
        scaling_coef_range = [1 / args.num_tasks]
    elif args.specify_lambda != "None":
        scaling_coef_range = [args.specify_lambda]
    elif args.method.name in ["iso_c", "iso_cts", "TSVM"]:
        scaling_coef_range = np.linspace(0.0, 3.0, args.n_eval_points)[1:]
    else:
        scaling_coef_range = np.linspace(0.0, 1.0, args.n_eval_points)[1:]

    if args.method.name == "tall_mask":
        if args.method.load_mask:
            print("=" * 43, f"Evaluating the loaded TALL masks", "=" * 43)
            info["loaded_mask"] = evaluate_task_vector_at_coef(
                task_vector, pretrained_checkpoint, args, 1.0, eval_masks
            )
            print(
                "\t avg_normalized_top1: {}%\t avg_top1: {}%".format(
                    round(info["loaded_mask"]["avg_normalized_top1"] * 100, 2),
                    round(info["loaded_mask"]["avg_top1"] * 100, 2),
                )
            )
        else:
            for tall_mask_lambda in [0.2, 0.3, 0.4, 0.5, 0.6]:
                print("\n" * 2)
                print("=" * 43, f"tall_mask_lambda = {tall_mask_lambda:.2f}", "=" * 43)
                info[tall_mask_lambda] = evaluate_task_vector_at_coef(
                    task_vector,
                    pretrained_checkpoint,
                    args,
                    1.0,
                    eval_masks[tall_mask_lambda],
                )
                print(
                    "\t avg_normalized_top1: {}%\t avg_top1: {}%".format(
                        round(info[tall_mask_lambda]["avg_normalized_top1"] * 100, 2),
                        round(info[tall_mask_lambda]["avg_top1"] * 100, 2),
                    )
                )
    else:
        best_avg_top1 = 0.0
        not_best_counter = 0
        for scaling_coef in scaling_coef_range:
            print("\n" * 2)
            print("=" * 43, f"alpha = {scaling_coef:.2f}", "=" * 43)
            info[scaling_coef] = evaluate_task_vector_at_coef(
                task_vector,
                pretrained_checkpoint,
                args,
                scaling_coef,
                eval_masks,
            )
            print(
                "\t avg_normalized_top1: {}%\t avg_top1: {}%".format(
                    round(info[scaling_coef]["avg_normalized_top1"] * 100, 2),
                    round(info[scaling_coef]["avg_top1"] * 100, 2),
                )
            )
            if args.early_stopping:
                if info[scaling_coef]["avg_top1"] > best_avg_top1:
                    best_avg_top1 = info[scaling_coef]["avg_top1"]
                    not_best_counter = 0
                else:
                    not_best_counter += 1
                    if not_best_counter >= args.early_stopping_patience:
                        print(f"Early stopping at alpha = {scaling_coef:.2f} due to no improvement in the last {args.early_stopping_patience} steps.")
                        break

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results
