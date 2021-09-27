from typing import List, Iterator, Dict, Tuple, Any
from overrides import overrides
import random
from copy import deepcopy

import numpy
from nltk import pos_tag
import torch

import torch.optim as optim
from torch.utils import data

from allennlp.data import Vocabulary
from allennlp.nn.util import move_to_device
from allennlp.interpret.attackers import Hotflip
from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers import Token
from allennlp.data import Batch
from allennlp.data.samplers import BucketBatchSampler


import allennlp.nn.util as nn_util
from allennlp.interpret.attackers.attacker import Attacker
import spacy
from allennlp.data.token_indexers import (
    ELMoTokenCharactersIndexer,
    TokenCharactersIndexer,
    SingleIdTokenIndexer,
)
from allennlp.interpret.attackers import utils
from allennlp.modules.token_embedders import Embedding
from attack_utils.model_util import evaluate_instances, evaluate_squad, \
    AllennlpDataset, get_gradients, get_integrated_gradients, \
    get_smooth_grads, get_embedding_matrix



class UniversalAttack():
    
    def __init__(self,model,
        vocab_namespace: str = "tokens", 
        distributed: bool = False) -> None:
        
        
        self._model = model
        self.cuda_device = next(self._model.named_parameters())[1].get_device()
        self.vocab = self._model.vocab
        self.namespace = vocab_namespace
        
        # Force new tokens to be alphanumeric
        self.invalid_replacement_indices: List[int] = []
        for i in self.vocab._index_to_token[self.namespace]:
            if not self.vocab._index_to_token[self.namespace][i].isalnum():
                self.invalid_replacement_indices.append(i)

        # for distributed
        self.distributed = distributed
        if self.distributed:
            from torch.nn import DataParallel
            self._model = DataParallel(self._model)


    def attack_squad(
        self,
        instances,
        test_instances, 
        init_triggers,  
        num_epoch, 
        mode,
        data_free,
        dataset_reader=None,
        max_tokens=None
        ):
        embedding_matrix: torch.Tensor = get_embedding_matrix(self._model, dataset_reader, self.namespace, max_tokens) 

        # initialize trigger tokens and metrix
        trigger_tokens = []
        update_idx = []
        answer_text = ''
        for i, token in enumerate(init_triggers.split(' ')):
            if token[0] == '{' and token[-1]=='}': # update triggers
                trigger_tokens.append(token[1:-1])
                update_idx.append(i)
            else: # target span
                trigger_tokens.append(token)
                answer_text += token + ' '
        answer_text = answer_text.rstrip()
    
        # initialize log for output
        log_trigger_tokens = [] 
        metrics_lst = [] 
        loss_lst = [] 
        
        trigger_ids = [self.vocab.get_token_index(token_txt, namespace=self.namespace) for token_txt in trigger_tokens]
        trigger_ids = [trigger_ids[i] for i in update_idx]
        
        for epoch in range(num_epoch):

            
            batch_copy = deepcopy(instances)
            # prepend triggers
            batch_prepended = prepend_batch(batch_copy, trigger_tokens=trigger_tokens, vocab=self.vocab, task='squad', tokenizer=dataset_reader._tokenizer)
            
            # update triggers
            # grad: [B, T, C]
            if mode == 'smooth':
                averaged_grad = get_smooth_grads(self._model, batch_prepended, cuda_device=self.cuda_device, task='squad', \
                                        span_start=4, span_end=5, answer_text=answer_text) 
            else:
                averaged_grad = get_gradients(self._model, batch_prepended, cuda_device=self.cuda_device, task='squad', \
                                        span_start=4, span_end=5, answer_text=answer_text, update_idx=update_idx) 
             
            # update
            new_trigger_ids = update_tokens(averaged_grad, trigger_ids, embedding_matrix, self.invalid_replacement_indices)
            trigger_tokens = []
            for new_trigger_id in new_trigger_ids:
                token_txt = self.vocab.get_token_from_index(new_trigger_id, namespace=self.namespace)
                trigger_tokens.append(token_txt)


            # evaluate new triggers on test
            if test_instances is not None:
                prepended_test_instances = prepend_batch(test_instances, trigger_tokens=trigger_tokens, vocab=self.vocab, task='squad', tokenizer=dataset_reader._tokenizer)
                pertubated_accuracy, pertubated_loss = evaluate_squad(prepended_test_instances, 
                                                    self._model, 
                                                    cuda_device=self.cuda_device, 
                                                    distributed=self.distributed, 
                                                    span_start=4,
                                                    span_end=5,
                                                    answer_text=answer_text)
            else:
                prepended_instances = prepend_batch(instances, trigger_tokens=trigger_tokens, vocab=self.vocab, task='squad', tokenizer=dataset_reader._tokenizer)
                pertubated_accuracy, pertubated_loss = evaluate_squad(prepended_instances, 
                                                    self._model, 
                                                    cuda_device=self.cuda_device, 
                                                    distributed=self.distributed, 
                                                    span_start=4,
                                                    span_end=5,
                                                    answer_text=answer_text)

            # record metrics for output
            log_trigger_tokens.append('-'.join( trigger_tokens))
            metrics_lst.append(pertubated_accuracy)
            loss_lst.append(pertubated_loss)


        return log_trigger_tokens, metrics_lst, loss_lst
                

    def attack_instances(
        self,
        instances: List[Instance],
        test_instances: List[Instance],
        init_triggers: str,
        target_label:int ,
        batch_size: int = 128,
        blacklist=List[Any],
        first_cls = False,
        num_epoch: int =1,
        patient: int = 10,
        mode = ''
        ) : 

        embedding_matrix: torch.Tensor = get_embedding_matrix(self._model) 

        # initialize trigger tokens and metrix
        trigger_tokens = []
        update_idx = []
            
        for i, token in enumerate(init_triggers.split(' ')):
            if token[0] == '{' and token[-1]=='}':
                trigger_tokens.append(token[1:-1])
                update_idx.append(i)
        if first_cls:
            update_idx = [idx+1 for idx in update_idx]

        pertubated_accuracy = None
        pertubated_loss = None

        # get target label id
        if str(target_label) in self.vocab._token_to_index['labels'].keys():
            target_label_id = self.vocab._token_to_index['labels'][str(target_label)]
        
        # add blacklist 
        for token in blacklist:
            if token in self.vocab._token_to_index[self.namespace].keys():
                idx = self.vocab._token_to_index[self.namespace][token]
                self.invalid_replacement_indices.append(idx)
  
        
        
        
        # initialize log for output
        log_trigger_tokens = [] 
        metrics_lst = [] 
        loss_lst = []
        
        # log for no triggers and initialized triggers
        orig_accuracy, orig_loss = evaluate_instances(test_instances, 
                                            self._model, 
                                            distributed=self.distributed, 
                                            target_label=target_label_id)
        prepended_test_instances = prepend_batch(test_instances, trigger_tokens=trigger_tokens, vocab=self.vocab)
        pertubated_accuracy, pertubated_loss = evaluate_instances(prepended_test_instances, 
                                            self._model,  
                                            distributed=self.distributed, 
                                            target_label=target_label_id)
                                            
                                            
    
        log_trigger_tokens.extend(['', '-'.join(trigger_tokens)])
        metrics_lst.extend([orig_accuracy, pertubated_accuracy])
        loss_lst.extend([orig_loss, pertubated_loss])
        
        
        # sample batches, update the triggers, and repeat
        idx_for_best = 0
        worst_accuracy = 1
        idx_so_far = 0
        dataset = AllennlpDataset(instances, self.vocab)
        for epoch in range(num_epoch):
            if idx_so_far - idx_for_best  >= patient:
                break

          
            

            batch_sampler = data.BatchSampler(data.SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
            for indices in batch_sampler:
                batch_copy = [dataset[i] for i in indices]
                if idx_so_far - idx_for_best  >= patient:
                    break
                    
                # prepend triggers
                batch_prepended = prepend_batch(batch_copy, trigger_tokens=trigger_tokens, vocab=self.vocab)
                
                # update triggers
                trigger_ids = [self.vocab.get_token_index(token_txt, namespace=self.namespace) for token_txt in trigger_tokens]
                # grad: [B, T, C]
                if mode == 'integrated':
                    grads = get_integrated_gradients(self._model, batch_prepended, target_label_id, ) 
                elif mode == 'smooth':
                    grads = get_smooth_grads(self._model, batch_prepended, target_label_id, ) 
                else:
                    grads = get_gradients(self._model, batch_prepended, target_label_id, ) 
                # average grad:  [T, C]
                averaged_grad = numpy.sum(grads, 0)[update_idx] 
                # update
                new_trigger_ids = update_tokens(averaged_grad, trigger_ids, embedding_matrix, self.invalid_replacement_indices)
                trigger_tokens = []
                for new_trigger_id in new_trigger_ids:
                    token_txt = self.vocab.get_token_from_index(new_trigger_id, namespace=self.namespace)
                    trigger_tokens.append(token_txt)


                # evaluate new triggers on test
                prepended_test_instances = prepend_batch(test_instances, trigger_tokens=trigger_tokens, vocab=self.vocab)
                pertubated_accuracy, pertubated_loss = evaluate_instances(prepended_test_instances, 
                                            self._model, 
                                            distributed=self.distributed, 
                                            target_label=target_label_id)
                             
                # if accuracy is worse
                if pertubated_accuracy <= worst_accuracy: 
                    worst_accuracy = pertubated_accuracy
                    idx_for_best = idx_so_far
                idx_so_far += 1

                # record metrics for output
                log_trigger_tokens.append('-'.join( trigger_tokens))
                metrics_lst.append(pertubated_accuracy)
                loss_lst.append(pertubated_loss)


        return log_trigger_tokens, metrics_lst, loss_lst
    
    def attack(
            self,
            instance: Instance,
            test_instances: List[Instance],
            init_triggers: str,
            target_label:int ,
            blacklist=List[Any],
            first_cls = False,
            num_epoch: int =1,
            patient: int = 10,
            mode = ''
        ) : 

        embedding_matrix: torch.Tensor = get_embedding_matrix(self._model,  ) 

        # initialize trigger tokens and metrix
        trigger_tokens = []
        update_idx = []
            
        for i, token in enumerate(init_triggers.split(' ')):
            if token[0] == '{' and token[-1]=='}':
                trigger_tokens.append(token[1:-1])
                update_idx.append(i)
        if first_cls:
            update_idx = [idx+1 for idx in update_idx]

        pertubated_accuracy = None
        pertubated_loss = None

        # get target label id
        if str(target_label) in self.vocab._token_to_index['labels'].keys(): # vocab responsible for label <-> label_id
            target_label_id = self.vocab._token_to_index['labels'][str(target_label)]
        # add blacklist 
        for token in blacklist:
            if token in self.vocab._token_to_index[self.namespace].keys():
                idx = self.vocab._token_to_index[self.namespace][token]
                self.invalid_replacement_indices.append(idx)
        
        
        
        
        # initialize log for output
        log_trigger_tokens = [] 
        metrics_lst = [] 
        loss_lst = []
        
        # log for no triggers and initialized triggers
        orig_accuracy, orig_loss = evaluate_instances(test_instances, 
                                            self._model, 
                                            distributed=self.distributed, 
                                            target_label=target_label_id)
        prepended_test_instances = prepend_batch(test_instances, trigger_tokens=trigger_tokens, vocab=self.vocab)
        pertubated_accuracy, pertubated_loss = evaluate_instances(prepended_test_instances, 
                                            self._model,  
                                            distributed=self.distributed, 
                                            target_label=target_label_id)
                                            
                                            
    
        log_trigger_tokens.extend(['', '-'.join(trigger_tokens)])
        metrics_lst.extend([orig_accuracy, pertubated_accuracy])
        loss_lst.extend([orig_loss, pertubated_loss])
        
        
        # sample batches, update the triggers, and repeat
        idx_for_best = 0
        worst_accuracy = 1
        idx_so_far = 0
        
        for epoch in range(num_epoch):
            if idx_so_far - idx_for_best  >= patient:
                break

            # prepend triggers
            instance_prepended = prepend_instance(instance, trigger_tokens=trigger_tokens, vocab=self.vocab)
            
            # update triggers
            trigger_ids = [self.vocab.get_token_index(token_txt, namespace=self.namespace) for token_txt in trigger_tokens]
            # grad: [B, T, C]
            if mode == 'integrated':
                grads = get_integrated_gradients(self._model, [instance_prepended], target_label_id, ) 
            elif mode == 'smooth':
                grads = get_smooth_grads(self._model, [instance_prepended], target_label_id, ) 
            else:
                grads = get_gradients(self._model, [instance_prepended], target_label_id, ) 
            # average grad:  [T, C]
            averaged_grad = numpy.sum(grads, 0)[update_idx] 
            # update
            new_trigger_ids = update_tokens(averaged_grad, trigger_ids, embedding_matrix, self.invalid_replacement_indices)
            trigger_tokens = []
            for new_trigger_id in new_trigger_ids:
                token_txt = self.vocab.get_token_from_index(new_trigger_id, namespace=self.namespace)
                trigger_tokens.append(token_txt)


            # evaluate new triggers on test
            prepended_test_instances = prepend_batch(test_instances, trigger_tokens=trigger_tokens, vocab=self.vocab)
            pertubated_accuracy, pertubated_loss = evaluate_instances(prepended_test_instances, 
                                        self._model, 
                                        distributed=self.distributed, 
                                        target_label=target_label_id)
                            
            # if accuracy is worse
            if pertubated_accuracy <= worst_accuracy: 
                worst_accuracy = pertubated_accuracy
                idx_for_best = idx_so_far
            idx_so_far += 1

            # record metrics for output
            log_trigger_tokens.append('-'.join( trigger_tokens))
            metrics_lst.append(pertubated_accuracy)
            loss_lst.append(pertubated_loss)
            


        return log_trigger_tokens, metrics_lst, loss_lst    

def update_tokens(grad, token_ids, embedding_matrix, invalid_replacement_indices):

    new_token_ids = [None] * len(token_ids)
    # pass the gradients to a particular attack to generate substitute token for each token.
    for index_of_token_to_flip in range(len(token_ids)):
        
        # Get new token using taylor approximation.
        new_token_id = _first_order_taylor(
            grad[index_of_token_to_flip, :], 
            torch.from_numpy(numpy.array(token_ids[index_of_token_to_flip])),
            embedding_matrix= embedding_matrix,
            invalid_replacement_indices = invalid_replacement_indices
        )

        new_token_ids[index_of_token_to_flip] = new_token_id 
    return new_token_ids


def prepend_batch(instances, trigger_tokens=None, vocab=None, task=None, tokenizer=None):
    """
    trigger_tokens List[str] ：
    """
    instances_with_triggers = []
    for instance in deepcopy(instances): 
        instance_perturbed = prepend_instance(instance, trigger_tokens, vocab, task, tokenizer)
        instances_with_triggers.append(instance_perturbed)
    
    return instances_with_triggers

def prepend_instance(instance, trigger_tokens, vocab, task=None, tokenizer=None):

    if task is None: # text classification
        if str(instance.fields['tokens'].tokens[0]) == '[CLS]':
            instance.fields['tokens'].tokens = [instance.fields['tokens'].tokens[0]] + \
                [Token(token_txt) for token_txt in trigger_tokens] + \
                instance.fields['tokens'].tokens[1:]
        else:
            instance.fields['tokens'].tokens = [Token(token_txt) for token_txt in trigger_tokens] + instance.fields['tokens'].tokens
        instance.fields['tokens'].index(vocab)
        return instance
    elif task == 'squad':
        instance.fields['passage'].tokens = [Token(token_txt) for token_txt in trigger_tokens] + instance.fields['passage'].tokens
        instance.fields['passage'].index(vocab)
    
        # add triggers to metadata
        instance.fields['metadata'].metadata['original_passage'] = ' '.join(trigger_tokens) + " " + instance.fields['metadata']['original_passage']
        instance.fields['metadata'].metadata['passage_tokens'] = trigger_tokens + instance.fields['metadata'].metadata['passage_tokens']

        # update passage_offsets
        passage_tokens = tokenizer.tokenize(instance.fields['metadata']['original_passage'])
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        instance.fields['metadata'].metadata['token_offsets'] = passage_offsets
        return instance


# this is totally same as HotFlip. (I put this method here for myself convenient checking)
def _first_order_taylor(grad: numpy.ndarray, token_idx: torch.Tensor, embedding_matrix:  torch.Tensor, invalid_replacement_indices, cuda_device=0) -> int:
    """
    The below code is based on
    https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/
    research/adversarial/adversaries/brute_force_adversary.py

    Replaces the current token_idx with another token_idx to increase the loss. In particular, this
    function uses the grad, alongside the embedding_matrix to select the token that maximizes the
    first-order taylor approximation of the loss.
    we want to minimize (x_perturbed-x) * grad(L_adv)
    
    """
    grad = nn_util.move_to_device(torch.from_numpy(grad), cuda_device)

    word_embedding = torch.nn.functional.embedding(
        nn_util.move_to_device(torch.LongTensor([token_idx]), cuda_device),
        embedding_matrix,
    )
    word_embedding = word_embedding.detach().unsqueeze(0)
    grad = grad.unsqueeze(0).unsqueeze(0)
    # solves equation (3) here https://arxiv.org/abs/1903.06620
    new_embed_dot_grad = torch.einsum("bij,kj->bik", (grad, embedding_matrix)) # each instance: grad shape (seq_len, emb_size) ;emb_matrix shape (vocab_size, emb_size)
    prev_embed_dot_grad = torch.einsum("bij,bij->bi", (grad, word_embedding)).unsqueeze(-1)
    dir_dot_grad = new_embed_dot_grad - prev_embed_dot_grad # minimize (x_perturbed-x) * grad(L_adv)
    dir_dot_grad = dir_dot_grad.detach().cpu().numpy()   
    neg_dir_dot_grad = - dir_dot_grad  # maximize -(x_perturbed-x) * grad(L_adv)
    # Do not replace with non-alphanumeric tokens
    neg_dir_dot_grad[:, :, invalid_replacement_indices] = -numpy.inf
    best_at_each_step = neg_dir_dot_grad.argmax(2)

    
    return best_at_each_step[0].data[0]
        

