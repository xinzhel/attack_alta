Code for the paper [Exploring the Vulnerability of Natural Language Processing Models via Universal Adversarial Texts](https://alta2021.alta.asn.au/files/ALTW_2021_paper_13.pdf)

We use `allennlp` and the supplement library `allennlp_models` as the boiler plate to organize our experiments. 


You should create `outputs`, `models`, `data` folders for saving the attack results, saving trained models and loading datasets.

# Train models
The entry point for training models is `main_train.py`. You could train models via bash commands. Or you could train all supported models via the script in `experiments/${task}/train.sh`.
```
# define task and models
$ task=bi_sst
$ model_type=bert_cnn

# define configuration path and path for saving models
$train_config_path=experiments/${task}/${model_type}.json
$serialization_dir=models/${task}/${model_type}

# remove previously-runed information
$rm -rf ${serialization_dir}


# train
$python -u main_train.py      $train_config_path              \
                    --serialization-dir $serialization_dir          \
                    --include-package my_library
```


# Attack
The entry point for attacking models is `attack_classifier.py`. You could attack models via bash commands. Or you could attack all supported models via the script in `experiments/${task}/attack_all_models.sh`.
```
$ task=bi_sst
$ model_type=bert_cnn
$ mode=integrated 
$ python -u attack_classifier.py       --model-type ${model_type}             \
                        --task ${task}                                        \
                        --mode ${mode}                                        \
                        --one-example                                                         
```

# Transfer Attack
The entry point for attacking models is `attack_transfer.py`. So far, you have to manually change the argument to perform the attack.
