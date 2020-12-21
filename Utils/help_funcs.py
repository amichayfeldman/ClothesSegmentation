import torch
import glob


def freeze_blocks(block):
    for child in block.children():
        if isinstance(child, torch.nn.Conv2d) or isinstance(child, torch.nn.BatchNorm2d):
            child.weight.requires_grad = False
            if child.bias is not None:
                child.bias.requires_grad = False
            else:
                freeze_blocks(child)


def instances_score(output, target, num_of_classes):
    '''
    Function to computation of how model deals with each class.
    The output is vector with the ratio of predicted / target foreach class
    '''
    # Find which classes appear in the batch images
    num_of_classes_vec_target = torch.zeros(num_of_classes, dtype=torch.long).reshape(1, num_of_classes).repeat(
        target.shape[0], 1).cuda()
    num_of_classes_vec_predicted = torch.zeros(num_of_classes, dtype=torch.long).reshape(1, num_of_classes).repeat(
        output.shape[0], 1).cuda()
    for idx, target_img, output_img in zip(range(target.shape[0]), target, output):
        # for target:
        unique_elements_target, count_target = torch.unique(target_img, return_counts=True)
        num_of_classes_vec_target[idx, unique_elements_target.long()] = count_target

        # for output:
        unique_elements_predicted, count_prediction = torch.unique(output_img, return_counts=True)
        num_of_classes_vec_predicted[idx, unique_elements_predicted.long()] = count_prediction

    # Compute the ratios between predicted & target foreach class and mean per batch
    eps = 1e-5
    batch_class_histogram = torch.mean(num_of_classes_vec_predicted.type(torch.FloatTensor) /
                                       (num_of_classes_vec_target.type(torch.FloatTensor) + eps), dim=0)

    batch_class_histogram[batch_class_histogram >= eps] = 0.1
    batch_class_histogram[batch_class_histogram == 0] = 1
    instances_score = 1.0 / (batch_class_histogram + eps)
    return batch_class_histogram, num_of_classes_vec_predicted, num_of_classes_vec_target

