#!/bin/bash

# command line arguments
cd ../../
task='bi_sst'
model_type='ft_cnn'
train_config_path=experiments/${task}/${model_type}.json
serialization_dir=models/${task}/${model_type}

# remove previously-runed information
rm -rf ${serialization_dir}


# apply unlearnable and train
python -u main_train.py      $train_config_path              \
                    --serialization-dir $serialization_dir          \
                    --include-package my_library