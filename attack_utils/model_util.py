from typing import List, Iterator, Dict, Tuple, Any
import os
import glob
from torch.utils.data import Dataset, Sampler, BatchSampler, SequentialSampler
import torch
from torch import backends
import pandas as pd
import numpy as np
import itertools

from allennlp.data.instance import Instance
from allennlp.data import Batch
from allennlp.nn import util as nn_util
from allennlp.data import Vocabulary
from allennlp.data import Batch
from allennlp.data.samplers import BucketBatchSampler
from allennlp.models import load_archive
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BertPooler, CnnEncoder, LstmSeq2VecEncoder
from allennlp.predictors.predictor import Predictor

from transformers import BertTokenizer

from my_library.models import MyPooler
from my_library.models.sst_classifier import SstClassifier
from my_library.modules import SelfAttentionEncoder


# Load the model and its associated vocabulary.
def load_local_archive(dir, task, model_type):       
    model_paths = glob.glob(f'{dir}{task}/{model_type}/*tar.gz')
    if not os.path.isfile(model_paths[0]):
        raise FileExistsError(f'{model_type} for {task} does not exist in the directory {dir}')
    model = load_archive(model_paths[0]).model

    return model


def _forward(input_tensor_dict, model, 
    distributed, cuda_device='cuda:0'):
    model.to(cuda_device)
    model.get_metrics(reset=True)
    # forward pass
    if distributed:
        outputs = model(**input_tensor_dict)
    else:
        outputs = model(**input_tensor_dict)

    torch.cuda.empty_cache() # in case of OOM error
    if distributed:
        outputs['loss'] = outputs['loss'].mean()
        return outputs, model.module
    else:
        return outputs, model

def evaluate_instances(instances, model, distributed=False, 
    return_just_loss=True, target_label=None):
    """  inference
    """
    
    # input tensor
    dataset = Batch(instances)
    dataset.index_instances(model.vocab)
    input_tensor_dict = dataset.as_tensor_dict()
    if target_label is not None:
        input_tensor_dict['label'] = torch.empty(input_tensor_dict['label'].shape, 
                                                dtype=input_tensor_dict['label'].dtype, 
                                                device=input_tensor_dict['label'].device).fill_(target_label)
    cuda_device = next(model.named_parameters())[1].get_device()
    input_tensor_dict = nn_util.move_to_device(input_tensor_dict, cuda_device)
    # forward
    output_dict, model_no_dist = _forward(input_tensor_dict, model, distributed)


    # return accuracy and model output (logits and loss)
    if return_just_loss:
        return model_no_dist.get_metrics(reset=True)['accuracy'], float(output_dict['loss'])
    else:
        return model_no_dist.get_metrics(reset=True)['accuracy'], output_dict

def filter_instances_by_prediction(instances, model):
    batch_sampler = BatchSampler(SequentialSampler(instances), batch_size=32, drop_last=False)
    batches = (
            [instances[i] for i in batch_indices]
            for batch_indices in batch_sampler
        )
    total_correct = []
    for batch in  batches:
        # input tensor
        dataset = Batch(batch)
        dataset.index_instances(model.vocab)
        input_tensor_dict = dataset.as_tensor_dict()
        cuda_device = next(model.named_parameters())[1].get_device()
        input_tensor_dict = nn_util.move_to_device(input_tensor_dict, cuda_device)
        with torch.no_grad():
            model.eval()
            logits = model(**input_tensor_dict)['logits']
        model.get_metrics(reset=True)
        predictions = logits.max(-1)[1].unsqueeze(-1)
        gold_labels = input_tensor_dict['label'].view(-1).long()
        # This is of shape (batch_size, ..., top_k).
        correct = predictions.eq(gold_labels.unsqueeze(-1)).squeeze().cpu().numpy().tolist()
        total_correct.extend(correct)
    return list(itertools.compress(instances, total_correct))


def evaluate_squad(instances, model, span_start, span_end, answer_text, distributed=False):
    """  inference
    """
    # evaluate in batch in case of large `instances`
    batch_sampler = BucketBatchSampler(batch_size=32, sorting_keys=["passage", "question"])
    batches = (
            [instances[i] for i in batch_indices]
            for batch_indices in batch_sampler.get_batch_indices(instances)
        )

    # metrics: F1 and EM with the targeted span
    total_f1 = 0.0
    total_em = 0.0
    total = 0.0
    cuda_device = next(model.named_parameters())[1].get_device()
    for batch in batches:
        # index for tensor
        dataset = Batch(batch)
        dataset.index_instances(model.vocab)
        input_tensor_dict = dataset.as_tensor_dict()
        # spans are set to the spot where the target inside the trigger is
        input_tensor_dict['span_start'] = torch.LongTensor([span_start]).repeat(input_tensor_dict['passage']['tokens']['tokens'].shape[0], 1).cuda()
        input_tensor_dict['span_end'] = torch.LongTensor([span_end]).repeat(input_tensor_dict['passage']['tokens']['tokens'].shape[0], 1).cuda()
        

        input_tensor_dict = nn_util.move_to_device(input_tensor_dict, cuda_device)
        
       
        with torch.no_grad():
            output_dict, model_no_dist = _forward(input_tensor_dict, model, distributed)
            for span_str in output_dict['best_span_str']:
                metrics = SquadEmAndF1()
                metrics.get_metric(reset=True)
                metrics(span_str, [answer_text])
                em, f1 = metrics.get_metric()
                total_f1 += f1
                total_em += em
                total += 1.0
    return total_em / total, float(output_dict['loss'])


def get_gradients(model, instances, \
    target_label=None, task=None, \
    span_start=None, span_end=None, answer_text=None,update_idx=None):

    # register backward hook
    embedding_gradients: List[Tensor] = []

    def hook_layers(module, grad_in, grad_out):
        grads = grad_out[0]
        embedding_gradients.append(grads)

    hook_handles = []
    embedding_layer = nn_util.find_embedding_layer(model)
    handle = embedding_layer.register_full_backward_hook(hook_layers)
    hook_handles.append(handle)

    original_param_name_to_requires_grad_dict = {}
    for param_name, param in model.named_parameters():
        original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
        param.requires_grad = True


    cuda_device = next(model.named_parameters())[1].get_device()
    if task is None: # text classification
        assert target_label is not None
        # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
        with backends.cudnn.flags(enabled=False):
            dataset = Batch(instances)
            dataset.index_instances(model.vocab)
            input_tensor_dict = dataset.as_tensor_dict()
            # TODO: use Adv loss explicitely rather than use training loss
            input_tensor_dict['label'] = torch.empty(input_tensor_dict['label'].shape, 
                                                        dtype=input_tensor_dict['label'].dtype, 
                                                        device=input_tensor_dict['label'].device).fill_(target_label)
            input_tensor_dict = nn_util.move_to_device(input_tensor_dict, cuda_device)
            loss = model.forward(**input_tensor_dict)['loss']  # type: ignore
            # grad_fn = torch.nn.CrossEntropyLoss()
            # loss = grad_fn(outputs['logits'], input_tensor_dict['label'])
            loss.backward()
        grads = embedding_gradients[0].detach().cpu().numpy()
    elif task == "squad": # question answering
        assert span_start is not None 
        assert span_end is not None 
        # evaluate in batch in case of large `instances`
        batch_sampler = BucketBatchSampler(batch_size=32, sorting_keys=["passage", "question"])
        batches = (
                [instances[i] for i in batch_indices]
                for batch_indices in batch_sampler.get_batch_indices(instances)
            )

        batch_count = 0
        for batch in batches:
            # index for tensor
            dataset = Batch(batch)
            dataset.index_instances(model.vocab)
            input_tensor_dict = dataset.as_tensor_dict()
            # spans are set to the spot where the target inside the trigger is
            input_tensor_dict['span_start'] = torch.LongTensor([span_start]).repeat(input_tensor_dict['passage']['tokens']['tokens'].shape[0], 1).cuda()
            input_tensor_dict['span_end'] = torch.LongTensor([span_end]).repeat(input_tensor_dict['passage']['tokens']['tokens'].shape[0], 1).cuda()
            

            input_tensor_dict = nn_util.move_to_device(input_tensor_dict, cuda_device)
            
            # return loss for backward to get gradient        
            # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
            with backends.cudnn.flags(enabled=False):
                loss = _forward(input_tensor_dict, model, )[0]['loss']
                loss.backward()
            if batch_count == 0 :
                # average grad:  [T, C]
                grads = torch.sum(embedding_gradients[0], dim=0).detach().cpu().numpy()[update_idx] # embed grad for the paragraph
                embedding_gradients = []
            else:
                grads += torch.sum(embedding_gradients[0], dim=0).detach().cpu().numpy()[update_idx]
                embedding_gradients = []
            batch_count = batch_count + 1
        grads = grads / batch_count

        # Zero gradients.
        # NOTE: this is actually more efficient than calling `self._model.zero_grad()`
        # because it avoids a read op when the gradients are first updated below.
        for p in model.parameters():
            p.grad = None

        

    for hook in hook_handles:
        hook.remove()

    # restore the original requires_grad values of the parameters
    for param_name, param in model.named_parameters():
        param.requires_grad = original_param_name_to_requires_grad_dict[param_name]
    return grads


def get_smooth_grads(model, instances, \
        target_label=None,  \
        stdev = 0.001, num_samples = 10 , \
        span_start=None, span_end=None, answer_text=None):
    
    total_gradients = None

    def _register_forward_hook( stdev: float):
        
        def forward_hook(module, inputs, output):
            # Random noise = N(0, stdev * (max-min))
            scale = output.detach().max() - output.detach().min()
            noise = torch.randn(output.shape, device=output.device) * stdev * scale

            # Add the random noise
            output.add_(noise)

        # Register the hook
        embedding_layer = nn_util.find_embedding_layer(model)
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    for _ in range(num_samples):
        handle = _register_forward_hook(stdev)
        try:
            grads = get_gradients(model, instances, target_label,)
        finally:
            handle.remove()

        # Sum gradients
        if total_gradients is None:
            total_gradients = grads
        else:
            total_gradients += grads

    # Average the gradients
    total_gradients /= num_samples

    return total_gradients


def get_integrated_gradients(model, instances, \
    target_label=None):

    ig_grads = None

    # Use 10 terms in the summation approximation of the integral in integrated grad
    steps = 10

    def _register_hooks(model, alpha: int):
        def forward_hook(module, inputs, output):
            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hooks
        handles = []
        embedding_layer = nn_util.find_embedding_layer(model)
        handles.append(embedding_layer.register_forward_hook(forward_hook))
        return handles

    # Exclude the endpoint because we do a left point integral approximation
    for alpha in np.linspace(0, 1.0, num=steps, endpoint=False):
        handles = []
        # Hook for modifying embedding value
        handles = _register_hooks(model, alpha)
        grads = get_gradients(model, instances, target_label)
        for handle in handles:
            handle.remove()
        # Running sum of gradients
        if ig_grads is None:
            ig_grads = grads
        else:
            ig_grads += grads

        # Average of gradient term
        ig_grads /= steps
        
        return ig_grads


def get_input_score(grads, embeddings):
    """

    # Parameters

        grads : `numpy.ndarray` shape (bsz, seq_len, embsize)
        embeddings : `numpy.ndarray` shape (bsz, seq_len, embsize)

    # Returns

    """
    emb_product_grad = grads * embeddings # shape: (seq_len, embsize)
    aggregate_emb_product_grad = np.sum(emb_product_grad, axis=1) # shape: (seq_len)
    norm = np.linalg.norm(aggregate_emb_product_grad, ord=1) # 
    normalized_scores = [math.fabs(e) / norm for e in aggregate_emb_product_grad]

    return normalized_scores


def calculate_trigger_diversity(trigger_tokens=None, embedding_matrix=None, namespace=None, vocab=None):
        if len(trigger_tokens) == 1:
            return 0
        trigger_token_ids = [vocab.get_token_index(trigger_tokens[i], namespace=namespace) for i in range(len(trigger_tokens))]
        word_embedding = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids), embedding_matrix.cpu())
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        diversity_score = 1
        for i in range(len(trigger_tokens)):
            for j in range(i+1, len(trigger_tokens)):
                diversity_score *= cos(word_embedding[i], word_embedding[j])
        
        diversity_score = 1 - diversity_score
        return diversity_score.item()


def get_embedding_matrix( model, dataset_reader=None, namespace=None, max_tokens=None):
    if dataset_reader is None: # directly access 
        embedding_layer = nn_util.find_embedding_layer(model)
        assert isinstance(embedding_layer, (Embedding, torch.nn.modules.sparse.Embedding))
        # If we're using something that already has an only embedding matrix, we can just use
        # that and bypass this method.
        return embedding_layer.weight

    else: # for >1 embedding layers (e.g., BiDAF), we use the the code from allennlp to manually construct it in this case
        from allennlp.interpret.attackers.hotflip import Hotflip
        hotflip = Hotflip( predictor=Predictor(model, dataset_reader), vocab_namespace = namespace, max_tokens = max_tokens)
        return hotflip._construct_embedding_matrix()

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        random.seed(22)
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            select_class = self.keys[self.currentkey]
            select_idx_for_this_class = self.indices[self.currentkey]
            yield self.dataset[select_class][select_idx_for_this_class]
            # currectkey back to 0 if each calss has been selected
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx):
        return dataset.__getitem__(idx)['label']._label_id

    def __len__(self):
        return self.balanced_max*len(self.keys)


def get_attack_result(path, select_index=None, only_triggers=False, return_loss=False):
    result_df = pd.read_csv(path)[['iteration', 'triggers', 'accuracy', 'loss', 'diversity']]
    triggers_lst = list(result_df['triggers']) # only evaluate on the last one
    accuracy_lst = list(result_df['accuracy'])
    loss_lst = list(result_df['loss'])
    diversity_lst = list(result_df['diversity'])
    if select_index is None:
        select_index = result_df['accuracy'].argmin()
    
    if select_index == 0:
        ttr = -1
    else:
        ttr = len(set(triggers_lst[select_index].split('-')))

    if only_triggers:
        return triggers_lst[select_index]
    if return_loss:
        return triggers_lst[select_index], accuracy_lst[select_index], loss_lst[select_index], diversity_lst[select_index], ttr
    else:
        return triggers_lst[select_index], accuracy_lst[select_index], diversity_lst[select_index], ttr


def filter_instance_by_label(instances, label_filter, exclude=True):
        targeted_instances = []
        # not _label_id
        assert type(label_filter) == type(instances[0]['label'].label)
        for instance in instances:
            if exclude:
                if instance['label'].label != label_filter:
                    targeted_instances.append(instance)
            else:
                if instance['label'].label == label_filter:
                    targeted_instances.append(instance)
        return targeted_instances


class AllennlpDataset(Dataset):
    """
    An `AllennlpDataset` is created by calling `.read()` on a non-lazy `DatasetReader`.
    It's essentially just a thin wrapper around a list of instances.
    """

    def __init__(self, instances: List[Instance], vocab: Vocabulary = None):
        self.instances = instances
        self.vocab = vocab

    def __getitem__(self, idx) -> Instance:
        if self.vocab is not None:
            self.instances[idx].index_fields(self.vocab)
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def __iter__(self) -> Iterator[Instance]:
        """
        Even though it's not necessary to implement this because Python can infer
        this method from `__len__` and `__getitem__`, this helps with type-checking
        since `AllennlpDataset` can be considered an `Iterable[Instance]`.
        """
        yield from self.instances

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab
