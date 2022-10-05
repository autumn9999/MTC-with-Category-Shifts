import torch
from utils import *

def experiment(data_loaders, config):
    num_iter = config["num_iter"]
    test_interval = config["test_interval"]
    from general_model import general_model
    model = general_model(config, config["task_num"], config["class_num"])
    len_renew = min([len(loader) - 1 for loader in data_loaders["train"]])
    file_out = config["file_out"]
    log_name = config["log_name"]

    best_average_acc_list_unseen= [0.0 for i in range(config["task_num"])]
    best_average_acc_list_seen= [0.0 for i in range(config["task_num"])]
    best_H_list= [0.0 for i in range(config["task_num"])]
    best_iter_num = 0

    for iter_num in range(1, num_iter + 1):
        # testing
        if iter_num % test_interval == 0 or iter_num == 1:
            if config["missing_rate"] < 1.0:
                _, average_acc_list_unseen = test_acc(config, data_loaders["test_unseen"], model)
            _, average_acc_list_seen = test_acc(config, data_loaders["test_seen"], model)

            # print and save models------------------------------------------
            if config["missing_rate"] < 1.0:
                average_acc_list_unseen = np.array(average_acc_list_unseen)
                average_acc_list_seen = np.array(average_acc_list_seen)
                H_list = (2 * average_acc_list_unseen * average_acc_list_seen) / (average_acc_list_unseen + average_acc_list_seen)
                log_string(file_out, 'Iter {:05d} Average Unseen/Seen/H Average Acc on all tasks, {:.2f}, {:.2f}, {:.2f}'.format(
                    iter_num,  np.average(average_acc_list_unseen),np.average(average_acc_list_seen), np.average(H_list)))
                for i in range(config["task_num"]):
                    log_string(file_out,"Iter {:05d} Unseen/Seen/H accuracy on Task {:d}, {:.2f}, {:.2f}, {:.2f}".format(
                        iter_num, i,
                        average_acc_list_unseen[i],
                        average_acc_list_seen[i],
                        H_list[i]))

                # update the best one
                if np.average(H_list) > np.average(best_H_list):
                    best_average_acc_list_unseen = average_acc_list_unseen
                    best_average_acc_list_seen = average_acc_list_seen
                    best_H_list = H_list
                    best_iter_num = iter_num

                    # saving models
                    save_dict = {}
                    save_dict["iter_num"] = best_iter_num
                    save_dict["model"] = model.model.state_dict()
                    save_dict["task_representation"] = model.model.task_representation
                    save_dict["class_prototype"] = model.model.class_prototype
                    torch.save(save_dict, log_name + "/best_model.pth.tar")

                # print best accuracy
                log_string(file_out, '******Best Average Unseen/Seen/H Average Acc on all tasks, {:.2f}, {:.2f}, {:.2f}'.format(
                    np.average(best_average_acc_list_unseen),np.average(best_average_acc_list_seen), np.average(best_H_list)))
                for i in range(config["task_num"]):
                    log_string(file_out, "Iter {:05d} Unseen/Seen/H accuracy on Task {:d}, {:.2f}, {:.2f}, {:.2f}".format(
                        best_iter_num, i,
                        best_average_acc_list_unseen[i],
                        best_average_acc_list_seen[i],
                        best_H_list[i]))

            else:
                average_acc_list_seen = np.array(average_acc_list_seen)
                average_acc = np.mean(average_acc_list_seen)

                # print current accuracy
                log_string(file_out, 'Current Avg accur is obtained on Iter {:05d}: {:.2f}'.format(iter_num, average_acc))
                for i in range(config["task_num"]):
                    log_string(file_out,"Iter {:05d} Seen accuracy on Task {:d}, {:.2f}".format(
                                   iter_num, i, average_acc_list_seen[i]))

                # update the best one
                if np.average(average_acc_list_seen) > np.average(best_average_acc_list_seen):
                    best_average_acc_list_seen = average_acc_list_seen
                    best_iter_num = iter_num

                    # saving models
                    save_dict = {}
                    save_dict["iter_num"] = best_iter_num
                    save_dict["model"] = model.model.state_dict()
                    save_dict["task_representation"] = model.model.task_representation
                    save_dict["class_prototype"] = model.model.class_prototype
                    torch.save(save_dict, log_name + "/best_model.pth.tar")

                # print best accuracy
                log_string(file_out,'Best Avg accuracy is obtained on Iter {:05d}: {:.2f}'.format(best_iter_num, np.average(best_average_acc_list_seen)))
                for i in range(config["task_num"]):
                    log_string(file_out,"Iter {:05d} Seen accuracy on Task {:d}, {:.2f}".format(
                                   best_iter_num, i, best_average_acc_list_seen[i]))

            log_string(file_out, "\n")
        #------------------------------------------------------------------------------------

        # training
        if (iter_num-1) % len_renew == 0:
            iter_list = [iter(loader) for loader in data_loaders["train"]]
        inputs_batch = []
        labels_batch = []

        for iter_ in iter_list:
            data_list = iter_.next()
            image = data_list[0].cuda()
            labels = data_list[1].cuda()

            inputs_batch.append(image)
            labels_batch.append(labels)
        model.one_train_iteration(inputs_batch, labels_batch)

def test_acc(config, loaders, model):
    average_acc_list = []
    class_acc_list = []

    iter_val = [iter(loader) for loader in loaders]
    for i in range(len(iter_val)):
        iter_ = iter_val[i]
        start_test = True
        class_acc = []
        for j in range(len(loaders[i])):
            inputs, labels = iter_.next()
            inputs = inputs.float().cuda()
            labels = labels.cuda()
            predicts, labels = model.one_test_iteration(inputs, labels, i)
            if start_test:
                all_predict = predicts.float().detach()
                all_label = labels.float().detach()
                start_test = False
            else:
                all_predict = torch.cat((all_predict, predicts.float().detach()), 0)
                all_label = torch.cat((all_label, labels.float().detach()), 0)

        for i in range(config["class_num"]):
            class_index = torch.squeeze(all_label).float() == i
            label_number_current_class = torch.sum(class_index)
            correct_number_current_class = torch.sum(all_predict[class_index] == i)
            if label_number_current_class == 0:
                class_acc.append(None)
            else:
                acc = correct_number_current_class.item() / float(label_number_current_class.item())
                class_acc.append(acc*100)
        class_acc_list.append(class_acc)

        # average_acc
        right_number = torch.sum(torch.squeeze(all_predict).float() == all_label)
        average_acc = right_number.item() / float(all_label.size()[0])
        average_acc_list.append(average_acc*100)
    return class_acc_list, average_acc_list
