import torch
import numpy as np
import matplotlib.pyplot as plt
import configparser

from Utils.help_funcs import instances_score
from Dataset.dataset import get_dataloaders


def evaluate(model, test_dataloader, num_of_classes):
    with torch.no_grad():
        test_loss_list = [None] * len(test_dataloader)
        test_running_loss = 0.0
        model.eval()

        n_rows = 4 * 5 + 1
        n_cols = 3
        f, axarr = plt.subplots(n_rows, n_cols, constrained_layout=False)
        f.set_size_inches(100, 200)

        # fig, ax = plt.subplot(n_rows, n_cols)
        width = 0.2
        sample_idx = 1
        total_prediction_class_hist = torch.zeros(1, num_of_classes).cuda()
        total_target_class_hist = torch.zeros(1, num_of_classes).cuda()

        for idx, test_data in enumerate(test_dataloader):
            if idx == 5:
                break
            inputs, labels, reg_gt_map = test_data['image'].to(device), test_data['gt'].to(device), test_data[
                'gt_reg_map'].to(device)
            outputs = model(inputs.type(torch.FloatTensor).to(device))
            loss = focal_loss(outputs['out'].cuda(), reg_gt_map.squeeze().cuda().long())
            prediction = torch.nn.functional.softmax(outputs['out'], dim=1)
            test_running_loss += loss.item()
            test_loss_list[idx] = loss.item()

            predicted_seg_map = torch.argmax(prediction, dim=1).squeeze()
            instace_loss_vec, prediction_hist, target_hist = instances_score(output=predicted_seg_map.cuda(),
                                                                             target=test_data['gt_reg_map'].cuda())
            total_prediction_class_hist += prediction_hist.sum(dim=0)
            total_target_class_hist += target_hist.sum(dim=0)

            for s in range(inputs.shape[0]):
                # pdb.set_trace()
                axarr[sample_idx, 0].imshow(predicted_seg_map[s, ...].detach().cpu().numpy().astype(np.uint8),
                                            cmap='hot')
                unnormalized_img = (255 * (inputs[s, ...].detach().cpu().permute(1, 2, 0).numpy() + 1) / 2).astype(
                    np.int32)
                axarr[sample_idx, 1].imshow(unnormalized_img.astype(np.uint8))
                axarr[sample_idx, 2].bar(np.arange(1, num_of_classes) - width / 2,
                                         prediction_hist[s, 1:].detach().cpu().numpy(), width, label='prediction')
                axarr[sample_idx, 2].bar(np.arange(1, num_of_classes) + width / 2,
                                         target_hist[s, 1:].detach().cpu().numpy(), width, label='target')
                axarr[sample_idx, 2].title.set_text("Instances hist")
                axarr[sample_idx, 2].set_xticks(np.arange(1, num_of_classes))
                axarr[sample_idx, 2].set_xlabel("Class")
                axarr[sample_idx, 2].set_ylabel("Num of occurance[mean per batch]")
                axarr[sample_idx, 2].legend()
                sample_idx += 1
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0001, hspace=0.25)
        plt.show()

        test_loss = test_running_loss / (idx + 1)
    print('test loss = {}'.format(test_loss))
    print("Classes probabilities - prediction:")
    print(total_prediction_class_hist / sample_idx)
    print("Classes probabilities - target:")
    print(total_target_class_hist / sample_idx)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    test_dataloader = get_dataloaders(config=config)

    best_model_path = glob.glob(os.path.join(output_folder, '*BEST*'))
    if len(best_model_path) > 0:
        deepLab_V3_pretrained.load_state_dict(torch.load(best_model_path[0]))
        deepLab_V3_pretrained.eval()

        evaluate(model=deepLab_V3_pretrained, test_dataloader=test_dataloader)