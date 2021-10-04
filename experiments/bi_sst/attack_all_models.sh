#!/bin/bash
cd ../../
task='bi_sst'
for model_type in bert_cnn bert_attention; do 
	
    echo "Attack ${model_type}."
    
    # attack
     python -u attack_classifier.py --model-type ${model_type} --task ${task}                                         \

    python -u attack_classifier.py       --model-type ${model_type}             \
                        --task ${task}                                          \
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

