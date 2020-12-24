#!/usr/bin/env python

config = {
    'Paths': {
        "data_folder": '../clothes_data/',
        "output_folder": './runnings/Dec_24',
        "model_checkpoint":''
        },

    'Params': {
        "batch_size": 4,
        "lr": 1e-3,
        "wd": 1e-6,
        "epochs": 200,
        "target_size": (256, 256),
        "num_of_classes": 59
    }
}
