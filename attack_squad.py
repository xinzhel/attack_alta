import argparse
import os
from attack_utils.data_util import load_data, get_squad_dataset_reader
from attack_utils.model_util import load_local_archive
from attack_utils.universal_attack import UniversalAttack
import pandas as pd

def main(task, model_type, data_free=False, distributed=False, \
            grad=None, num_epoch=1, max_samples=None, \
            pretrained_transformer=None):

    # load model
    model = load_local_archive(dir='models/', task=task, model_type=model_type)
    model.to(device='cuda:0')
    # # load 100 test_data, 360 dev data for "who" questions
    dataloader_dev = load_data(task, 'dev', pretrained_transformer=pretrained_transformer)
    who_questions = []
    for item in dataloader_dev.iter_instances():
        for word in item['question']:
            if word.text.lower() == 'who':
                who_questions.append(item)
                break
        if max_samples is not None and len(who_questions) > max_samples:
            break

    dev_data = who_questions
    # test_data = who_questions[360:]
    # print(f'Size of dev data: {len(dev_data)}; \n Size of test data: {len(test_data)}')
    


    # init trigger
    trigger_init = "{the} {the} {the} {the} donald trump {the} {the} {the} {the}"
    file_name = f'outputs/squad/{model_type}'
    if data_free:
        file_name += '_df'
    if grad is not None:
        file_name += '_'+grad

    if not os.path.isfile(file_name+'.csv'):
        # attack 
        universal = UniversalAttack(model, 
                        vocab_namespace='tokens',
                        distributed=distributed)
        
        log_trigger_tokens, metrics_lst, loss_lst = universal.attack_squad(
                dev_data,
                test_instances=None, 
                dataset_reader=get_squad_dataset_reader(),
                init_triggers=trigger_init,  
                num_epoch=num_epoch, 
                mode=grad,
                data_free=data_free
            )
        # save the result
        result_df = pd.DataFrame(
            {
                'triggers': [ele for ele in log_trigger_tokens], \
                "accuracy": [ele for ele in metrics_lst], \
                "loss":  [ele for ele in loss_lst]
                
            }
        )
        result_df.to_csv(file_name+'.csv')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type", 
        type=str, 
        default='bidaf',
        help="path to the victim model"
    )

    parser.add_argument(
        "--task", 
        type=str, 
        default='squad',
        help="path to the victim model"
    )


    parser.add_argument(
        "--num-epoch", 
        type=int, 
        default=100, 
        help="the number of epoch"
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="",
    )
    parser.add_argument(
        "--data-free",
        action="store_true",
        default=False,
        help="",
    )

    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=None, 
        help=""
    )

    # Now we can parse the arguments.
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    
    args = parse_args()
    main(args.task, args.model_type, args.distributed, args.data_free, num_epoch=args.num_epoch, max_samples=args.max_samples)