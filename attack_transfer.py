import os
import sys
import argparse
import logging
from copy import deepcopy
import torch

from attack_utils.data_util import load_data, filter_instance_by_length
from attack_utils.model_util import load_local_archive, filter_instance_by_label, filter_instances_by_prediction
from attack_utils.universal_attack import UniversalAttack, prepend_batch, evaluate_instances

logging.basicConfig(filename='outputs/transfer.txt', level=logging.CRITICAL)

def transfer(trigger_tokens_lst, task, target_label, min_length=10,distributed=False):
    model_types = ['lstm', 'cnn', 'attention' ]

    #  'ft_lstm', 'ft_cnn', 'ft_attention', \
        # 'glove_lstm', 'glove_cnn', 'glove_attention', \
        # 'bert_lstm', 'bert_cnn', 'bert_attention'

    

    trigger_tokens_lst = [trigger_tokens.split('-') for trigger_tokens in trigger_tokens_lst]
        
    
    for model_type in model_types:
        # load model
        model = load_local_archive(dir='models/', task=task, model_type=model_type)
        model.to(device=torch.device('cuda:0'))

        # get some info to deal with data
        pretrained_transformer = None
        if 'bert' in model_type:
            pretrained_transformer = 'bert-base-cased'

        vocab_namespace = 'tokens'
        first_cls = False
        if pretrained_transformer:
            vocab_namespace = 'tags'
            first_cls = True


        # load 1280 test_data, 872 dev data
        if task == 'bi_sst':
            # 1280 filtered from 1821 test_data
            dataloader_test = load_data(task, 'test', pretrained_transformer=pretrained_transformer, distributed=distributed)
            test_data = list(dataloader_test.iter_instances())
            test_data = [test_data[i] for i in filter_instance_by_length(task, min_length=min_length)]

        elif task == 'imdb' or task == 'amazon' or task == 'yelp':
            num_test = 128 #1280
            num_dev = 87  # 872
            # sample seed is set in `ImdbDatasetReader._read`
            dataloader_test = load_data(task, 'test', pretrained_transformer=pretrained_transformer, distributed=distributed,
                        sample=num_test+num_dev,min_length=min_length, max_length=512-3) #lst_num_trigger_tokens[-1],
            test_data = list(dataloader_test.iter_instances())
            
            attack_data = test_data[num_test:(num_test+num_dev)]
            test_data = test_data[:num_test]
        elif task == 'ag_news':
            data_iterator = load_data(task, 'test', shuffle=True, pretrained_transformer=pretrained_transformer, distributed=distributed,).iter_instances()
            attack_data = list(itertools.islice(data_iterator, 320))
            test_data = list(itertools.islice(data_iterator, 0, 960))

        test_data = filter_instance_by_label(test_data, label_filter=target_label, exclude=True)
        test_data = filter_instances_by_prediction(test_data, model)
 
        # get target label id
        assert str(target_label) in model.vocab._token_to_index['labels'].keys()
        target_label_id = model.vocab._token_to_index['labels'][str(target_label)]
        
        logging.critical(f'Transfer to {model_type}' )
        for trigger_tokens in trigger_tokens_lst:
            prepended_test_instances = prepend_batch(test_data, trigger_tokens=trigger_tokens, vocab=model.vocab)
            accuracy, _ = evaluate_instances(prepended_test_instances, 
                                                model,  
                                                target_label=target_label_id)

            logging.critical('\t Triggers:' + '-'.join(trigger_tokens))
            logging.critical('\t Transfer accuracy is: ' + str(round(accuracy, 2)))
                
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", 
        type=str, 
        default='bi_sst',
    )

    # parser.add_argument(
    #     "--trigger-tokens", 
    #     type=str, 
    #     default='',
    # )

    parser.add_argument(
        "--target-label", 
        type=str, 
        default='0',
    )

    # Now we can parse the arguments.
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    
    args = parse_args()

    transfer([ 'graves-graves-graves', 'smothered-smothered-smothered', 'salient-salient-salient', 'ricture-ricture-ricture', 'Romething-Romething-Romething'],
     args.task, 
     args.target_label, 
     min_length=10,distributed=False)


    #  ag_news: '', 'druthers-thrift-insurance','Coinstar-TrimTabs-TrimTabs'




    
    


