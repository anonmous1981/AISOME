#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-06-01 13:37:43.576184507 +0530
# Modify: 2022-06-14 12:47:29.718800476 +0530

"""Training and evaluation for BertMultiLabel"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from data_generator import BertMultiLabelDataset
from evaluate import evaluate
from metrics import metrics
from model.net import BertMultiLabel

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def train_one_epoch(model, optimizer, loss_fn, data_loader, params,
                    metrics, target_names, args):

    # Set model to train
    model.train()

    criterion = loss_fn

    # For the loss of each batch
    loss_batch = []
    accumulate = utils.Accumulate()

    # Training Loop
    for i, (data, target) in enumerate(iter(data_loader)):
        logging.info(f"Training on batch {i + 1}.")
        target = target.to(args.device)
        # Data is moved to relevant device in net.py after tokenization
        y_pred = model(data)
        loss = criterion(y_pred.float(), target.float())
        loss.backward()

        # Sub-batching behaviour to prevent memory overload
        if (i + 1) % params.update_grad_every == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())

        outputs_batch = (y_pred.data.cpu().detach().numpy()
                         > params.threshold).astype(np.int32)

        targets_batch = (target.data.cpu().detach().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)

        # For debugging purposes
        print(y_pred)

        del data
        del target
        del outputs_batch
        del targets_batch
        del y_pred
        torch.cuda.empty_cache()

    else:
        # Last batch
        if (i + 1) % params.update_grad_every != 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_batch.append(loss.item())

    outputs, targets = accumulate()

    summary_batch = {metric: metrics[metric](outputs, targets, target_names)
                     for metric in metrics}
    summary_batch["loss_avg"] = sum(loss_batch) * 1./len(loss_batch)

    return summary_batch


def train_and_evaluate(model, optimizer, loss_fn, train_loader,
                       test_loader, params, metrics, exp_dir, name, args,
                       target_names, restore_file=None):
    # Default start epoch
    start_epoch = 0
    # Best train and test macro f1 variables
    best_train_macro_f1 = 0.0
    best_test_macro_f1 = 0.0

    # Load from checkpoint if any
    if restore_file is not None:
        restore_path = os.path.join(exp_dir, f"{restore_file}.pth.tar")

        logging.info(f"Found checkpoint at {restore_path}.")

        start_epoch = utils.load_checkpoint(restore_path, model, optimizer) + 1

    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"Logging for epoch {epoch}.")

        _ = train_one_epoch(model, optimizer, loss_fn, train_loader,
                            params, metrics, target_names, args)

        test_stats = evaluate(model, loss_fn, test_loader,
                              params, metrics, args, target_names)

        train_stats = evaluate(model, loss_fn, train_loader,
                               params, metrics, args, target_names)

        # Getting f1 test_stats

        train_macro_f1 = train_stats['f1']['macro_f1']
        is_train_best = train_macro_f1 >= best_train_macro_f1

        test_macro_f1 = test_stats['f1']['macro_f1']
        is_test_best = test_macro_f1 >= best_test_macro_f1

        logging.info(
                (f"Test macro F1: {test_macro_f1:0.5f}\n"
                 f"Train macro F1: {train_macro_f1:0.5f}\n"
                 f"Avg test loss: {test_stats['loss_avg']:0.5f}\n"
                 f"Avg train loss: {train_stats['loss_avg']:0.5f}\n"))

        # Save test_stats
        train_json_path = os.path.join(
                exp_dir, "metrics", f"{name}", "train",
                f"epoch_{epoch + 1}_train_f1.json")
        utils.save_dict_to_json(train_stats, train_json_path)

        test_json_path = os.path.join(
                exp_dir, "metrics", f"{name}", "test",
                f"epoch_{epoch + 1}_test_f1.json")
        utils.save_dict_to_json(test_stats, test_json_path)

        # Saving best stats
        if is_train_best:
            best_train_macro_f1 = train_macro_f1
            train_stats["epoch"] = epoch + 1

            best_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "train",
                    "best_train_f1.json")
            utils.save_dict_to_json(train_stats, best_json_path)

        if is_test_best:
            best_test_macro_f1 = test_macro_f1
            test_stats["epoch"] = epoch + 1

            best_json_path = os.path.join(
                    exp_dir, "metrics", f"{name}", "test",
                    "best_test_f1.json")
            utils.save_dict_to_json(test_stats, best_json_path)

            logging.info(
                    (f"New best macro F1: {best_test_macro_f1:0.5f} "
                     f"Train macro F1: {train_macro_f1:0.5f} "
                     f"Avg test loss: {test_stats['loss_avg']} "
                     f"Avg train loss: {train_stats['loss_avg']}."))

        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            }

        utils.save_checkpoint(state, is_test_best,
                              os.path.join(exp_dir, "model_states", f"{name}"),
                              (epoch + 1) % params.save_every == 0)

    # For the last epoch

    utils.save_checkpoint(state, is_test_best,
                          os.path.join(exp_dir, "model_states", f"{name}"),
                          True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dirs", nargs="+", type=str,
                        default=["data/"],
                        help=("Directory containing training "
                              "and testing cases."))
    parser.add_argument("-t", "--targets_paths", nargs="+", type=str,
                        default=["targets/targets.json"],
                        help="Path to target files.")
    parser.add_argument("-x", "--exp_dir", default="experiments/",
                        help=("Directory to load parameters "
                              " from and save metrics and model states"))
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="Name of model")
    parser.add_argument("-p", "--params", default="params.json",
                        help="Name of params file to load from exp+_dir")
    parser.add_argument("-de", "--device", type=str, default="cuda",
                        help="Device to train on.")
    parser.add_argument("-id", "--device_id", type=int, default=0,
                        help="Device ID to run on if using GPU.")
    parser.add_argument("-r", "--restore_file", default=None,
                        help="Restore point to use.")
    parser.add_argument("-ul", "--unique_labels", type=str, default=None,
                        help="Labels to use as targets.")
    parser.add_argument("-bm", "--bert_model_name", type=str,
                        default="bert-base-uncased",
                        help="BERT variant to use as model.")

    args = parser.parse_args()

    # Setting logger
    utils.set_logger(os.path.join(args.exp_dir, f"{args.name}.log"))

    # Selecting correct device to train and evaluate on
    if not torch.cuda.is_available() and args.device == "cuda":
        logging.info("No CUDA cores/support found. Switching to cpu.")
        args.device = "cpu"

    if args.device == "cuda":
        args.device = f"cuda:{args.device_id}"

    logging.info(f"Device is {args.device}.")

    # Loading parameters
    params_path = os.path.join(args.exp_dir, "params", f"{args.params}")
    assert os.path.isfile(params_path), f"No params file at {params_path}"
    params = utils.Params(params_path)

    # Setting seed for reproducability
    torch.manual_seed(47)
    if "cuda" in args.device:
        torch.cuda.manual_seed(47)

    # Setting data paths
    train_paths = []
    test_paths = []
    for path in args.data_dirs:
        train_paths.append(os.path.join(path, "train"))
        test_paths.append(os.path.join(path, "test"))

    # Datasets
    train_dataset = BertMultiLabelDataset(
                                    data_paths=train_paths,
                                    targets_paths=args.targets_paths,
                                    unique_labels=args.unique_labels)

    test_dataset = BertMultiLabelDataset(
                                    data_paths=test_paths,
                                    targets_paths=args.targets_paths,
                                    unique_labels=args.unique_labels)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                              shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=params.batch_size,
                             shuffle=True)

    model = BertMultiLabel(labels=train_dataset.unique_labels,
                           device=args.device,
                           hidden_size=params.hidden_dim,
                           max_length=params.max_length,
                           bert_model_name=args.bert_model_name,
                           truncation_side=params.truncation_side)

    model.to(args.device)

    # Defining optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.BCELoss(reduction='sum')

    train_and_evaluate(model, optimizer, loss_fn, train_loader,
                       test_loader, params, metrics, args.exp_dir,
                       args.name, args, train_dataset.unique_labels,
                       restore_file=args.restore_file)

    logging.info("="*80)


if __name__ == "__main__":
    main()
