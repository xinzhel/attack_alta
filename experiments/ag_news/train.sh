#!/bin/bash

# command line arguments
cd ../../
task='ag_news'
for model_type in lstm attention bert_lstm bert_cnn bert_attention ft_lstm ft_cnn ft_attention glove_lstm glove_cnn glove_attention; do
	if [ -e /home/xinzhel/git_repo/attack/experiments/ag_news/${model_type}.json ]; then
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


