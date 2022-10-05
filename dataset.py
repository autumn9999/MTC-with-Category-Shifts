import torch
import numpy as np
from PIL import Image
import os
import json
import data_augmentation

def make_dataset(dataset_task_path, class_split_2_specific_task, state, sample_state):
    '''
    Goal: collect the training or test samples path (from different classes) for each task
    Input: image_list, train_split (seen classes for each dataset each task),  state (for training or test)
    Output: list of (image, label)
    '''
    dataset = dataset_task_path.split("/")[-2]
    task = dataset_task_path.split("/")[-1]

    if dataset == "office-home":
        list_txt_path = './train_split/'+dataset
        task_index = 2
        class_index = 3

    sample_list = []
    if state == "training":
        train_split_path = list_txt_path + "_train.txt"
        train_sample_path = open(train_split_path).readlines()
        for sample in train_sample_path:
            sample_task = sample.split("/")[task_index]
            sample_class = sample.split("/")[class_index]
            if sample_task == task:   # !!! for the current task
                sample_class = sample_task+"/"+sample_class
                if sample_class in class_split_2_specific_task:  # train_class_split
                    sample_path = "../../" + sample.split()[0]
                    sample_list.append((sample_path, int(sample.split()[1])))

    elif state == "test":
        test_split_path = list_txt_path + "_test.txt"
        test_sample_path = open(test_split_path).readlines()
        for sample in test_sample_path:
            sample_task = sample.split("/")[task_index]
            sample_class = sample.split("/")[class_index]

            if sample_state == "unseen":
                if sample_task == task:
                    sample_class = sample_task+"/"+sample_class
                    if sample_class not in class_split_2_specific_task:
                        sample_path = "../../" + sample.split()[0]
                        sample_list.append((sample_path, int(sample.split()[1])))

            elif sample_state == "seen":
                if sample_task == task:
                    sample_class = sample_task + "/" + sample_class
                    if sample_class in class_split_2_specific_task:
                        sample_path = "../../" + sample.split()[0]
                        sample_list.append((sample_path, int(sample.split()[1])))
    return sample_list

def default_loader(path):
    '''
    Goal: load the image from its path, and tranform it from image to numpy
    '''
    img = Image.open(path)
    return img.convert('RGB')

class dataset(object):
    def __init__(self, dataset_task_path, missing_rate, state, sample_state="None"):

        dataset = dataset_task_path.split("/")[-2]
        task = dataset_task_path.split("/")[-1]
        class_split_dic = torch.load('class_split/' + dataset + '_{}_split.pt'.format(missing_rate))

        class_split =[]
        for i in range(len(class_split_dic)):
            class_split_one_task = []
            class_split_one_task.extend(class_split_dic[i])
            class_split.append(class_split_one_task)

        for i in range(len(class_split)):
            if class_split[i][0].split("/")[0] == task:
                class_split_2_specific_task = class_split[i]

        self.sample_list = make_dataset(dataset_task_path, class_split_2_specific_task, state, sample_state)
        print(task+"_" + state +"_"+ sample_state + " has {:d} samples.".format(len(self.sample_list)))
        if state == "training":
            self.transform = data_augmentation.transform_train(resize_size=256, crop_size=224)
        else:
            self.transform = data_augmentation.transform_test(resize_size=256, crop_size=224)
        self.loader = default_loader

    def __getitem__(self, index):
        path, target = self.sample_list[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.sample_list)