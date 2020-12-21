import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def save_model_checkpoint(model, epoch, output_path, best=False):
    # pdb.set_trace()
    if best:
        previous_best_pt = glob.glob(os.path.join(output_path, '*BEST.pt'))
        if len(previous_best_pt) > 0:
            os.remove(previous_best_pt[0])
        name = os.path.join(output_path, 'model_state_dict_epoch={}_BEST.pt'.format(epoch))
    else:
        name = os.path.join(output_path, 'model_state_dict_epoch={}.pt'.format(epoch))
    torch.save(model.state_dict(), name)


def plot_current_results(total_epochs, train_loss_list, val_loss_list):
    plt.figure()
    plt.plot(np.arange(len(train_loss_list)), train_loss_list, label='train loss', color='r')
    plt.plot(np.arange(len(train_loss_list)), val_loss_list, label='val loss', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim([0, 3])
    plt.xlim([0, total_epochs])
    plt.show()


def train_model(model, data_loaders_dict, config, losses_list, save_model=True, write_csv=False):
    lr = config.getfloat('Params', 'lr')
    wd = config.getfloat('Params', 'wd')
    alpha = config.getfloat('Params', 'alpha')
    epochs = config.getint('Params', 'epochs')
    output_folder = config['Paths']['output_folder']
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', "Cuda is not working"
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    train_dataloader, val_dataloader = data_loaders_dict['train_dl'], data_loaders_dict['val_dl']

    train_loss_list, val_loss_list, lr_list, wd_list = [], [], [], []
    best_train_loss, best_val_loss = np.inf, np.inf
    if not os.path.isdir(os.path.join(output_folder, 'saved_checkpoints')):
        os.makedirs(os.path.join(output_folder, 'saved_checkpoints'))

    for epoch in range(epochs):
        running_loss = 0.0
        # --- TRAIN:  --- #
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, labels, reg_gt_map = data['image'].to(device), data['gt'].to(device), data['gt_reg_map'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.type(torch.FloatTensor).to(device))
            dice_loss_ = dice_loss(outputs['out'].cuda(), labels.squeeze().cuda().long())
            # ce_loss = crossentr_loss(outputs['out'].cuda(), reg_gt_map.squeeze().cuda().long())
            focalloss_value = focal_loss(outputs['out'].cuda(), reg_gt_map.squeeze().cuda().long())
            loss = alpha * focalloss_value + (1 - alpha) * dice_loss_
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / (i + 1)
        train_loss_list.append(train_loss)
        ##################

        model.eval()
        # --- VAL:  --- #
        with torch.no_grad():
            val_running_loss = 0.0
            for val_i, val_data in enumerate(val_dataloader):
                inputs, labels, reg_gt_map = val_data['image'].to(device), val_data['gt'].to(device), val_data[
                    'gt_reg_map'].to(device)
                outputs = model(inputs.type(torch.FloatTensor).to(device))
                if outputs['out'].shape != labels.squeeze().shape:
                    print("output shape:{}, label shape:{}".format(outputs['out'].shape, labels.squeeze().shape))
                dice_loss_ = dice_loss(outputs['out'].cuda(), labels.squeeze().cuda().long())
                # ce_loss = crossentr_loss(outputs['out'].cuda(), reg_gt_map.cuda().long())
                focalloss_value = focal_loss(outputs['out'].cuda(), reg_gt_map.squeeze().cuda().long())
                loss = 0.8 * focalloss_value + 0.2 * dice_loss_
                val_running_loss += loss.item()
            val_loss = val_running_loss / (val_i + 1)
            val_loss_list.append(val_loss)
        ##################

        # --- Save results to csv ---#
        lr_list.append(optimizer.param_groups[-1]['lr'])
        wd_list.append(optimizer.param_groups[-1]['weight_decay'])
        ##############################

        # --- Save model checkpoint ---#
        if val_loss < best_val_loss:
            best_model = model
            best_val_loss = val_loss
            if save_model:
                save_model_checkpoint(model=model, epoch=epoch,
                                      output_path=output_folder, best=True)
        elif epoch % 10 == 0:
            if save_model:
                save_model_checkpoint(model=model, epoch=epoch, output_path=output_folder, best=False)
        ##############################
        if scheduler is not None:
            scheduler.step(val_loss)

        if epoch > 15:
            train_dataloader.dataset.change_crop_size()
            val_dataloader.dataset.change_crop_size()

        if epoch % 10 == 0 and epoch > 0:
            plot_current_results(epoch, train_loss_list, val_loss_list)

        print("Epoch {}:  train loss: {:.5f}, val loss: {:.5f}".format(epoch, train_loss, val_loss))

    if write_csv:
        write_to_csv(os.path.join(output_path, 'results.csv'), [list(range(epochs)), train_loss_list, val_loss_list,
                                                                lr_list, wd_list])