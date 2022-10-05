# Association Graph Learning for Multi-Task Classification with Category Shifts
Code for paper “Association Graph Learning for Multi-Task Classification with Category Shifts” accepted to NeurIPS2022.

## Set Up
### Prerequisites
 - Python 3.9.7
 - Pytorch 1.11.0
 - GPU: an NVIDIA Tesla V100
 
### Getting Started
Inside this repository, we mainly conduct comprehensive experiments on Office-Home. Download the dataset from the following link, and place it in [`../../dataset/`](./dataset/). 
To split documents are obtained by randomly selecting 80% of samples from each task as the complete training set and use the remaining samples as the test set. 
The split documents used for the office-home dataset is provided in [`train_split/`](./train_split/).
The class assignment documents with different missing rates (75%, 50%, 25% and 0%) is provided in the tables of the supplemental materials.

- Office-home; [[link]](https://www.hemanthdv.org/officeHomeDataset.html)

## Experiments

To train the proposed association graph by running the command:
```
python set_up.py --gpu_id 0 --dataset office-home --missing_rate 0.75 --nlayers 4 
```
