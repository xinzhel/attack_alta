#!/bin/bash
cd ../../
task='yelp'
for model_type in bert_cnn bert_lstm bert_attention cnn attention ft_cnn ft_lstm ft_attention glove_cnn glove_lstm glove_attention; do
	
    echo "Attack ${model_type}."
    
    # attack
    python -u attack_classifier.py       --model-type ${model_type}             \
                        --task ${task}
                        --one-example 

    python -u attack_classifier.py       --model-type ${model_type}             \
                        --task ${task}                \
                        --mode integrated          \
                        --one-example 

    python -u attack_classifier.py       --model-type ${model_type}             \
                        --task ${task}                \
                        --mode smooth          \
                        --one-example 
	

done	

