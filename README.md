# attack

```
$ task=bi_sst
$ model_type=bert_cnn
$ mode=integrated 
$ python -u attack_classifier.py       --model-type ${model_type}             \
                        --task ${task}                                        \
                        --mode ${mode}                                        \
                        --one-example                                       
                        
```
