from allennlp_models.classification import *
from allennlp_models.rc import *
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, WhitespaceTokenizer, CharacterTokenizer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.dataset_readers import TextClassificationJsonReader
from my_library.dataset_readers import ImdbDatasetReader, \
        AmazonDatasetReader, \
        YelpDatasetReader


def filter_instance_by_length(task, min_length=10):
    dataloader_test = load_data(task, 'test', )
    data = list(dataloader_test.iter_instances())
    
    result = []
    
    for i, instance in enumerate(data):
        txt_field = instance.fields['tokens']
        if len(txt_field.tokens) >= min_length:
            result.append(i)
        if len(result) >= 1280:
            break
    print(f'Original number:{len(data)}; Final number: {len(result)}' )
    return result

def load_data(task, split, batch_size=64, shuffle = False, pretrained_transformer=None, max_length=None, min_length=None, 
                sample=None, distributed=False, num_worker=0):
     
    """ load data with its tokenizers and indexers
    Args:
        MODEL_TYPE: used to indicate whether using pretrained tokenizer and indexer
        max_length: for imdb
        min_length: 
        sample: for imdb
    """
    
    if distributed :
        manual_distributed_sharding = True
        manual_multiprocess_sharding = True
    else:
        manual_distributed_sharding = False
        manual_multiprocess_sharding = False

    # construct dataset reader
    if pretrained_transformer:
        tokenizer = PretrainedTransformerTokenizer(model_name=pretrained_transformer, max_length=max_length)
        indexer = PretrainedTransformerIndexer(model_name=pretrained_transformer) 
    else:
        tokenizer = None # use default tokenizer in `DataReader` 
        indexer = SingleIdTokenIndexer(lowercase_tokens=False) # word tokenizer
    if task == 'bi_sst':
        granularity = '2-class'
        reader = StanfordSentimentTreeBankDatasetReader(granularity=granularity, 
                                        tokenizer=tokenizer, token_indexers={"tokens": indexer},
                                        manual_distributed_sharding = manual_distributed_sharding,
                                        manual_multiprocess_sharding = manual_multiprocess_sharding,
                                        ) #distributed=distributed
        assert split in ['dev', 'train', 'test']
        file_path = f'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/{split}.txt'
    elif task == 'imdb':
        reader = ImdbDatasetReader(tokenizer=tokenizer, token_indexers={"tokens": indexer}, 
                                        sample_files=sample, min_len=min_length,
                                        manual_distributed_sharding = manual_distributed_sharding,
                                        manual_multiprocess_sharding = manual_multiprocess_sharding,
                                        distributed=distributed)
        assert split in [ 'train', 'test']
        file_path = split
    elif task == 'amazon':
        reader = AmazonDatasetReader(tokenizer=tokenizer, token_indexers={"tokens": indexer}, 
                                        sample=sample, min_len=min_length,
                                        manual_distributed_sharding = manual_distributed_sharding,
                                        manual_multiprocess_sharding = manual_multiprocess_sharding,
                                        distributed=distributed)
        assert split in [ 'train', 'test']
        file_path = split
    elif task == 'yelp':
        reader = YelpDatasetReader(tokenizer=tokenizer, token_indexers={"tokens": indexer}, 
                                        sample=sample, min_len=min_length,
                                        manual_distributed_sharding = manual_distributed_sharding,
                                        manual_multiprocess_sharding = manual_multiprocess_sharding,
                                        distributed=distributed)
        assert split in [ 'train', 'test']
        file_path = split
    elif task == 'squad': # for bidaf
        # SQuAD Reader is forced to read in parallel
        reader = get_squad_dataset_reader()
        assert split in ['dev', 'train']
        file_path = f'https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-{split}-v1.1.json'
        
    elif task == 'ag_news':
        reader = TextClassificationJsonReader()
        assert split in ['train', 'test']
        file_path = f"data/ag_news/{split}.json"

    dataloader = MultiProcessDataLoader(reader, file_path, batch_size=batch_size, num_workers=num_worker, shuffle=shuffle, max_instances_in_memory=64)
    
    
    return dataloader

def get_squad_dataset_reader():
    indexer = {
        "tokens": SingleIdTokenIndexer(lowercase_tokens=True),
        # "token_characters": TokenCharactersIndexer(
        #     character_tokenizer= CharacterTokenizer(byte_encoding='utf-8',end_tokens=[260], start_tokens = [259]), 
        #     min_padding_length=5),
        }
    reader = SquadReader(tokenizer=None, token_indexers=indexer)
    return reader