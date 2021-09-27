from attack_utils.data_util import load_data
import numpy as np
import os
from allennlp.data import Vocabulary
from allennlp.common.params import Params
import argparse

def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total # Pr(word) in row * Pr(class) in cloumn 
    oe = df / expected 
    return oe


def pmi(df, positive=True):
    df = observed_over_expected(df)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    args = parser.parse_args()
    # construct vocabulary
    vocab_dir = f'models/{args.task}/embedding--random__cnn/vocabulary/'
    dataloader = load_data(args.task, 'train', MODEL_TYPE='none')
    try:
        vocab = Vocabulary.from_files(vocab_dir)
        print('Load vocabulary.')
    except:
        print('No constructed vocabulary.')
        # vocab = Vocabulary.from_params(Params({}), instances=dataloader.iter_instances())
        # vocab.save_to_files(vocab_dir)

    # words x classes counting matrix
    file_name = f'outputs/{args.task}/count.npy'
    if not os.path.isfile(file_name):
        print('Construct words x classes counting matrix.')
        count_matrix = np.zeros(( len(vocab._index_to_token['tokens']),  len(vocab._index_to_token['label'])))
        dataloader.index_with(vocab)
        instances = dataloader.iter_instances() # iter after applying index
        for instance in instances:
            label_id = instance.fields['label']._label_id
            for indexed_token in instance.fields['tokens']._indexed_tokens['tokens']['tokens']:
                count_matrix[indexed_token][label_id] += 1
                # print(indexed_token)
        np.save(file_name, count_matrix)
    else:
        print('Load words x classes counting matrix.')
        count_matrix = np.load(file_name)

    # pmi
    pmi_df = pmi(count_matrix, positive=False)
    pmi_df = np.nan_to_num(pmi_df)

    # pmi rank
    neg_words_sorted_by_pmi = []
    pos_words_sorted_by_pmi = []
    for i in pmi_df[:,1].argsort()[::-1]: # 1 for negative class; 0 for positive class
        neg_words_sorted_by_pmi.append(vocab._index_to_token['tokens'][i])
    for i in pmi_df[:,0].argsort()[::-1]:
        pos_words_sorted_by_pmi.append(vocab._index_to_token['tokens'][i])

#     print('Top 10 words for positive class')
#     for i in pmi_df[:,0].argsort()[-10:]:
#         print(vocab._index_to_token['tokens'][i])
# 
#     print('Top 10 words for negative class')
#     for i in pmi_df[:,1].argsort()[-10:]:
#         print(vocab._index_to_token['tokens'][i])

    label_id = 1 # 1 for negative class; 0 for positive class
    pmi_rank = {1:neg_words_sorted_by_pmi, 0:pos_words_sorted_by_pmi}

    # get pmi rank for triggers
    for word in ['poky','Pompous', 'disgusting','unconstitutional'
    ,'nak']:
        try:
            token_rank = pmi_rank[label_id].index(word)
            freq_in_class = count_matrix[vocab._token_to_index['tokens'][word]][label_id] 
            total_freq = count_matrix[vocab._token_to_index['tokens'][word]][0] +  count_matrix[vocab._token_to_index['tokens'][word]][1]
            print(word, freq_in_class, '/', total_freq, token_rank)
        except:
            print(word + ' not in vocab')


    
    
