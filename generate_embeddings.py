#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-06-19 10:24:50.449259094 +0530
# Modify: 2022-06-19 12:51:17.641679477 +0530

import argparse
import logging
import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import utils
from data_generator import BertMultiLabelDataset
from model.net import BertMultiLabel

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def generate_embeddings(model, data_loader, params, args):

    if args.restore_file is not None:

        # Loading trained model
        logging.info(f"Found checkpoint at {args.restore_file}. Loading.")
        _ = utils.load_checkpoint(args.restore_file, model, device_id=0) + 1

    # Setting the model to evaluate
    model.eval()

    # For getting the embeddings

    # Generating embeddings
    for idx in range(len(data_loader)):
        doc, data = data_loader[idx]
        logging.info(f"Generating embeddings for {doc}.")

        y_pred, embed = model(data)

        # Removing from computation graph
        embed = embed.cpu().detach()

        # Saving the generated embeddings
        torch.save(embed, os.path.join(args.save_path, f"{doc}.pt"))

        del data
        del y_pred
        del embed
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dirs", nargs="+", type=str,
                        default=["data/"],
                        help=("Directory containing training "
                              "and testing cases."))
    parser.add_argument("-t", "--targets_paths", nargs="+", type=str,
                        default=["targets/targets.json"],
                        help="Path to targets json file. Give junk data.")
    parser.add_argument("-e", "--embed_path",
                        default="embeddings/embeddings.pkl",
                        help="Path to embeddings json file.")
    parser.add_argument("-x", "--exp_dir", default="experiments/",
                        help=("Directory to load parameters and to save "
                              "model states and metrics."))
    parser.add_argument("-p", "--params", default="params.json",
                        help="Name of params file to load from exp_dir.")
    parser.add_argument("-de", "--device", type=str, default="cuda",
                        help="Device to run on.")
    parser.add_argument("-r", "--restore_file", default=None,
                        help=("File to reload a saved model."))
    parser.add_argument("-n", "--name", type=str, default="gen_embeds",
                        help="Name of model.")
    parser.add_argument("-s", "--save_path", default="generated/", type=str,
                        help="Path to save generated embeddings.")
    parser.add_argument("-ul", "--unique_labels", type=str,
                        default=None, help="Labels to use as targets.")
    parser.add_argument("-bm", "--bert_model_name", type=str,
                        default="bert-base-uncased",
                        help="BERT variant to use as model.")

    args = parser.parse_args()

    # Creating save directory if it does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    utils.set_logger(os.path.join(args.save_path, f"{args.name}.log"))

    # Selecting the right device to train and evaluate on
    if not torch.cuda.is_available() and args.device == "cuda":
        logging.info("No CUDA cores/support found. Switching to cpu.")
        args.device = "cpu"

    logging.info(f"Device is {args.device}.")

    # Loading model parameters
    params_path = os.path.join(args.exp_dir, "params", f"{args.params}")
    assert os.path.isfile(
            params_path), f"No configuration file found at {params_path}."
    params = utils.Params(params_path)

    # Setting a seed for reproducability
    torch.manual_seed(47)
    if(args.device == "cuda"):
        torch.cuda.manual_seed(47)

    # Creating the training and testing data generators
    dataset = BertMultiLabelDataset(
                            data_paths=args.data_dirs,
                            targets_paths=args.targets_paths,
                            unique_labels=args.unique_labels,
                            mode="generate")

    model = BertMultiLabel(labels=dataset.unique_labels,
                           device=args.device,
                           hidden_size=params.hidden_dim,
                           max_length=params.max_length,
                           bert_model_name=args.bert_model_name,
                           truncation_side=params.truncation_side,
                           mode="generate")

    model.to(args.device)

    generate_embeddings(model=model,
                        data_loader=dataset,
                        params=params, args=args)


if __name__ == "__main__":
    main()
