import torch
import os
import argparse

from utils import *
import sys
from dataset import dataset
from main import experiment

project_path = os.getcwd()

if __name__ == "__main__":

    # Setup----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='multi-task classification with category shifts')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, nargs='?', default='office-home', help="dataset name")
    parser.add_argument('--log_name', type=str, nargs='?', default='logs/log-officehome', help="log name")
    parser.add_argument('--missing_rate', type=float, nargs='?', default=0.75, help="missing_rate")
    parser.add_argument('--num_iter', type=int, nargs='?', default=10000, help="defined the number of iterations")
    parser.add_argument('--nlayers', type=int, nargs='?', default=4, help="numbt of GNN layers")
    parser.add_argument('--beta', type=float, nargs='?', default=0.01, help="")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = {}

    config["gpu_id"] = args.gpu_id
    config["dataset"] = args.dataset
    config["log_name"] = args.log_name
    config["missing_rate"] = args.missing_rate
    config["num_iter"] = args.num_iter
    config["nlayers"] = args.nlayers
    config["beta"] = args.beta

    # fixed hyper-parameters
    config["lr"] = {"init_lr": 0.0001, "gamma": 0.5, "stepsize": 3000}
    os.system("mkdir -p "+ config["log_name"])
    config["file_out"] = open(config["log_name"] + "/train_log.txt", "w")

    config["basenet"] = 'resnet18'
    task_name_list = ["Art", "Clipart", "Product", "Real_World"]
    d_class = 65
    dataset_path = '../../dataset'
    config["test_interval"] = 100

    print(str(config) + '\n')
    config["file_out"].write(str(config) + '\n')
    st = ' '
    log_string(config["file_out"], st.join(sys.argv))

    config["task_num"] = len(task_name_list)
    config["class_num"] = d_class
    datasets = {"train": [], "test": []}
    batch_size = {"train": 32, "test": 32}
    config["batch_size"] = batch_size

    datasets["train"] = [dataset(dataset_path+"/"+config["dataset"]+"/"+task_name_list[i],
                        config["missing_rate"], "training") for i in range(len(task_name_list))]
    if config["missing_rate"] < 1.0:
        datasets["test_unseen"] = [dataset(dataset_path + "/" + config["dataset"] + "/" + task_name_list[i],
                                  config["missing_rate"], "test", "unseen") for i in range(len(task_name_list))]
    datasets["test_seen"] = [dataset(dataset_path + "/" + config["dataset"] + "/" + task_name_list[i],
                            config["missing_rate"], "test", "seen") for i in range(len(task_name_list))]

    data_loaders = {"train": [], "test_unseen": [], "test_seen": []}
    for train_dataset in datasets["train"]:
        data_loaders["train"].append(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size["train"], shuffle=True, num_workers=4))
    if config["missing_rate"] < 1.0:
        for test_unseen_dataset in datasets["test_unseen"]:
            data_loaders["test_unseen"].append(torch.utils.data.DataLoader(test_unseen_dataset, batch_size=batch_size["test"], shuffle=False, num_workers=4))
    for test_seen_dataset in datasets["test_seen"]:
        data_loaders["test_seen"].append(torch.utils.data.DataLoader(test_seen_dataset, batch_size=batch_size["test"], shuffle=False, num_workers=4))
    print("Data is ready to go !!!")

    experiment(data_loaders, config)
    print(config["log_name"])
    config["file_out"].close()
