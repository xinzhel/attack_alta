import os
import sys
import argparse
from copy import deepcopy
import torch
from nltk.corpus import opinion_lexicon
import pandas as pd
import itertools

from attack_utils.data_util import load_data, filter_instance_by_length
from attack_utils.model_util import load_local_archive, filter_instance_by_label, filter_instances_by_prediction
from attack_utils.universal_attack import UniversalAttack


def attack(task, model_type, trigger_tokens, targeted_class=None, num_epoch=4, vocab_namespace='tokens',  
    sst_granularity=2, min_length=10, 
    universal_perturb_batch_size=128, 
    distributed=False, first_cls=False, 
    mode='',
    one_example=False,
    pretrained_transformer=None,
    recover=True):

    # load model
    model = load_local_archive(dir='models/', task=task, model_type=model_type)
    model.to(device=torch.device('cuda:0'))
    # load test_data, attack data
    if task == 'bi_sst':
        # 1280 filtered from 1821 test_data
        dataloader_test = load_data(task, 'test', pretrained_transformer=pretrained_transformer, distributed=distributed)
        test_data = list(dataloader_test.iter_instances())
        test_data = [test_data[i] for i in filter_instance_by_length(task, min_length=min_length)]

        # attack with 872 attack_data
        dataloader_dev = load_data(task, 'dev', pretrained_transformer=pretrained_transformer, distributed=distributed)
        attack_data = list(dataloader_dev.iter_instances())
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

    test_data = filter_instances_by_prediction(test_data, model)

    if targeted_class is None:
        LABELS = list(model.vocab._token_to_index['labels'].keys())
    else:
        LABELS = [targeted_class]
        
    for target_label in LABELS:
        attack_data_c = filter_instance_by_label(attack_data, label_filter=target_label, exclude=True)
        test_data_c = filter_instance_by_label(test_data, label_filter=target_label, exclude=True)
        
        # no opinion lexicon as triggers
        if task in ['bi_sst', 'imdb', 'amazon', 'yelp']:
            blacklist = opinion_lexicon.negative() if target_label == '0' else opinion_lexicon.positive()
        else:
            blacklist = []
        
        file_name = f'outputs/{task}/{model_type}_{str(target_label)}{mode}.csv'
        if one_example:
            file_name = f'outputs/{task}/{model_type}_{str(target_label)}{mode}_one.csv'

        if os.path.isfile(file_name):
            if recover:
                print(f'Attack File {file_name} exist!')
                sys.exit()
            else:
                os.remove(file_name)
                print(f'Remove existing Attack File {file_name}!')

        # attack 
        universal = UniversalAttack(model, 
                        vocab_namespace=vocab_namespace,
                        distributed=distributed)
       
        if one_example:
            
            log_trigger_tokens, metrics_lst, loss_lst = universal.attack(attack_data_c[0],\
                test_instances=test_data_c, 
                init_triggers=trigger_tokens, 
                target_label=target_label, 
                blacklist=blacklist, 
                first_cls=first_cls, 
                num_epoch=num_epoch, 
                mode=mode
            )
        else:
            log_trigger_tokens, metrics_lst, loss_lst = universal.attack_instances(attack_data_c, \
                test_instances=test_data_c, 
                init_triggers=trigger_tokens, 
                target_label=target_label,
                batch_size = universal_perturb_batch_size,
                blacklist=blacklist, 
                first_cls=first_cls, 
                num_epoch=num_epoch, 
                mode=mode)
        
        # save the result
        result_df = pd.DataFrame(
            {
                "iteration": range(len([ele for ele in loss_lst ])), \
                'triggers': [ele for ele in log_trigger_tokens], \
                "accuracy": [ele for ele in metrics_lst ], \
                "loss":  [ele for ele in loss_lst ]
                
            }
        )
        print('Save to ', file_name)
        result_df.to_csv(file_name)
            
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type", 
        type=str, 
        default='lstm',
        # required=True,
        help="path to the victim model"
    )

    parser.add_argument(
        "--mode", 
        type=str, 
        default='', 
        # required=True,
        help=""
    )

    parser.add_argument(
        "--one-example",
        action="store_true",
        default=True,
        help="",
    )



    parser.add_argument(
        "--task", 
        type=str, 
        default='ag_news',
    )

    parser.add_argument(
        "--targeted-class", 
        type=str, 
        default=None,
    
    )

    parser.add_argument(
        "--num-epoch", 
        type=int, 
        default=None, 
        help="the number of epoch"
    )


    parser.add_argument(
        "--pretrained-transformer",
        type=str,
        default=None, #'bert-base-cased'
        help="",
    )


    parser.add_argument(
        "--opinion-lexicon-as-blacklist",
        action="store_true",
        default=True,
        help="constrain: filter out the sentiment words",
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="",
    )

    parser.add_argument(
        "--recover",
        action="store_true",
        default=False,
        help="",
    )

    # Now we can parse the arguments.
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    
    args = parse_args()
    torch.cuda.empty_cache() 

    # extra args
    
    trigger_tokens = '{the} {the} {the}'

    if args.task == "ag_news":
        args.targeted_class = '3'

    if args.num_epoch is None:
        args.num_epoch = 10 if args.one_example else 1

    # our experiment for PTM only trys bert so far
    if 'bert' in args.model_type:
        args.pretrained_transformer = 'bert-base-cased'

    vocab_namespace = 'tokens'
    first_cls = False
    if args.pretrained_transformer:
        vocab_namespace = 'tags'
        first_cls = True

    universal_perturb_batch_size=128
    if args.task == 'imdb' or args.task == 'yelp':
        # distributed = True
        universal_perturb_batch_size=8

    
    
    print(args.mode)
    attack(args.task, args.model_type,
        num_epoch=args.num_epoch, 
        vocab_namespace=vocab_namespace, 
        trigger_tokens=trigger_tokens, 
        targeted_class=args.targeted_class,
        universal_perturb_batch_size=universal_perturb_batch_size, 
        distributed=args.distributed,
        first_cls= first_cls,
        pretrained_transformer=args.pretrained_transformer,
        mode=args.mode,
        one_example=args.one_example,
        recover=args.recover)



