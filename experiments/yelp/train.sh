#!/bin/bash

# command line arguments
cd ../../
task='yelp'
for model_type in bert_cnn bert_attention  lstm ft_cnn ft_lstm ft_attention glove_cnn glove_lstm glove_attention; do
	if [ -e /data/zhanghe/xinzhe/attack/experiments/${task}/${model_type}.json ]; then
	       	echo "${model_type} exist."
			train_config_path=experiments/${task}/${model_type}.json
			serialization_dir=models/${task}/${model_type}

			# remove previously-runed information
			rm -rf ${serialization_dir}


			# apply unlearnable and train
			python -u main_train.py      $train_config_path              \
								--serialization-dir $serialization_dir          \
                    			--include-package my_library
	fi

done	

