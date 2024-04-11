# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower client example using PyTorch for CIFAR-10 image classification."""


import argparse
from torch_mist import estimate_mi
import timeit
from collections import OrderedDict
from sklearn.metrics import mutual_info_score
from importlib import import_module

import flwr as fl
import numpy as np
from time import time
import torch
import torchvision
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

import utils

# pylint: disable=no-member
#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

def normalize_data(data, data_min, data_max):
    """Normalize data to the range [data_min, data_max]."""
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    norm_data = norm_data * (data_max - data_min) + data_min
    return norm_data

def match_dimensions_and_calculate_mi(weights, inputs):
    """
    Uniformly sample elements from weights and inputs to match dimensions and calculate MI.
    sample_size: Number of elements to sample, ensuring it's less than min(input elements, weights)
    step: Step size to sample every 'step' elements from the flattened array.
    """
    # Flatten the inputs and weights
    inputs_flat = inputs.flatten()
    weights = np.array(weights).flatten()
    
    # Generate indices to sample every 'step' elements
    non_zero_weights = np.count_nonzero(weights)
    print(f"Number of non-zero weights: {non_zero_weights} out of {len(weights)}")
    
    # The number of samples to take from inputs is determined by the number of weights
    num_samples = len(weights)

    # Generate indices to sample from inputs
    indices = np.linspace(0, len(inputs_flat) - 1, num=num_samples, dtype=int)

    # Sample from inputs_flat using the generated indices
    inputs_sample = inputs_flat[indices]

    # Ensure weights_sample has the same length as inputs_sample
    weights_sample = weights[:num_samples]

    # Verify the sampling
    print(f"Number of indices: {len(indices)}")
    print(f"Input samples: {len(inputs_sample)}")
    print(f"Weight samples: {len(weights_sample)}")

    # Normalize inputs_sample to the range of weights_sample
    inputs_sample = normalize_data(inputs_sample, np.min(weights_sample), np.max(weights_sample))
    
    # Calculate mutual information
    # mi_estimate = mutual_info_score(inputs_sample, weights_sample)
    mi_estimate, log = estimate_mi(
            data=(inputs_sample, weights_sample),  
            estimator_name='js',  
            hidden_dims=[32, 32], 
            neg_samples=16,
            batch_size=128,
            max_epochs=500,
            valid_percentage=0.1,
            evaluation_batch_size=256,
            device='cpu',  
        )
    
    
    return mi_estimate

def get_weights(model: torch.nn.ModuleList) -> fl.common.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

class CifarClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        cid: str,
        model: torch.nn.Module,
        trainset: torchvision.datasets.CIFAR10,
        testset: torchvision.datasets.CIFAR10,
        device: torch.device,
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.device = device
        self.p = 1

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = get_weights(self.model)
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def _instantiate_model(self, model_str: str):

        # will load utils.model_str
        m = getattr(import_module("utils"), model_str)
        # instantiate model
        self.model = m()

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")
        # initial_weights = get_weights(self.model)

        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        pin_memory = bool(config["pin_memory"])
        num_workers = int(config["num_workers"])

        # fix_for_drop
        p = float(config["p"])
        if (p != self.p):
            print("changing p from " + str(self.p) + " to " + str(p))
            self.p = p
            self.model = utils.load_model("ResNet18", self.p)
            self.model.to(self.device)

        # Set model parameters
        set_weights(self.model, weights)

        if torch.cuda.is_available():
            kwargs = {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": True,
            }
        else:
            kwargs = {"drop_last": True}

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, **kwargs
        )
        t = time()
        inputs_list = []
        initial_weights, final_weights, inputs_array = utils.train(self.model, trainloader, epochs=epochs, device=self.device, inputs_list=inputs_list)

        # Calculate and flatten weight updates
        print("In training check:", np.count_nonzero(initial_weights), "non-zero elements in the last layer's weights.")
        fitTime = time() - t

        # Calculate mutual information
        # flat_inputs = np.concatenate(inputs_list).flatten()
        # print("flat_inputs.size", flat_inputs.size)     
        # print("weight_updates.size", initial_weights.size)

        # # Ensure the inputs and weight_updates are matched in dimensions
        # mi_estimate = match_dimensions_and_calculate_mi(initial_weights, flat_inputs)
        # print(f"Client {self.cid}: Mutual Information after fit round: {mi_estimate}")
        

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = get_weights(self.model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        metrics = {"duration": timeit.default_timer() - fit_begin}
        # metrics.update({"mutual_information": mi_estimate})
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics, fit_duration=fitTime
        )
        

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        set_weights(self.model, weights)
        set_weights(self.model, weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        loss, accuracy = utils.test(self.model, testloader, device=self.device)

        # Return the number of evaluation examples and the evaluation result (loss)
        metrics = {"accuracy": float(accuracy)}
        return EvaluateRes(
            num_examples=len(self.testset), loss=float(loss), metrics=metrics
        )


def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        required=True,
        help=f"gRPC server address",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where the dataset lives",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Net",
        choices=["Net", "ResNet18"],
        help="model to train",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="processor to run client on",
    )
    parser.add_argument(
        "--device_idx",
        type=int,
        default=0,
        help="processor to run client on",
    )


    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # model
    model = utils.load_model(args.model)
    if (args.device == "cpu"):
        device = torch.device("cpu", args.device_idx )
        print ("running on CPU")
    elif (torch.cuda.is_available()):
        print ("running on GPU")
        device = torch.device("cuda",args.device_idx)
    else:
        print ("GPU unavailble, running on CPU")
        device = torch.device("cpu", args.device_idx)
    model.to(device)
    # load (local, on-device) dataset
    trainset, testset = utils.load_dataset(args.model, args.cid)

    # Start client
    client = CifarClient(args.cid, model, trainset, testset, device)

    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()
