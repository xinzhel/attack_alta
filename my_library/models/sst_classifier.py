from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("sst_classifier")
class SstClassifier(Model):
    """
    This ``Model`` performs text classification for Stanford Sentiment Treebank. 

    The basic model structure: we'll embed the sentence, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    pass the vector through a feedforward network, the output of which we'll use as our scores for 
    each label.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    title_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the title to a vector.
    abstract_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the abstract to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 remove_cls: bool = False,
                 num_classes: int = 2,
                 attack: bool = False) -> None:
        super(SstClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.remove_cls = remove_cls
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder

        # for adv attack
        self.attack = attack
        self.origin_perturb_emb = None
        self.perturb_idx = None

        if classifier_feedforward is None:
            num_classes = vocab.get_vocab_size('labels') or num_classes
            self.classifier_feedforward = Linear(in_features=encoder.get_output_dim(),
                                      out_features=num_classes)
        else:
            self.classifier_feedforward = classifier_feedforward

        
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                #"accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

   

    @overrides
    def forward(self,  
                tokens: Dict[str, Dict[str, torch.LongTensor]],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        tokens : Dict[str, Dict[str, Variable]], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        
        tokens_mask = util.get_text_field_mask(tokens) # shape: (B, T)
        torch.cuda.empty_cache() # in case of OOM error
        embedded_tokens = self.text_field_embedder(tokens) # shape: (B, T, C)
        if self.remove_cls:
            embedded_tokens = embedded_tokens[:,1:,:]
            tokens_mask = tokens_mask[:, 1:]
        encoded_tokens = self.encoder(embedded_tokens, mask=tokens_mask) # shape: (B, hidden)
        
        logits = self.classifier_feedforward(encoded_tokens) # shape: (B, 2)
        # class_probabilities = F.softmax(logits, dim=-1)
        output_dict = {'logits': logits}
        if label is not None:
            for metric in self.metrics.values():
                metric(logits, label)
            
            output_dict["loss"] = self.loss(logits, label)
            # if self.attack:
            #     logits = torch.nn.Softmax()(logits)
                # uncertain loss
                # output_dict["loss"] = (logits[:,0] - logits[:,1]).abs().sum()
                # output_dict["loss"] = -((logits[:,0] - 0.5)**2 + (logits[:,1] - 0.5)**2).sum()
                # for adversarial attack, maximize cosine similarity
                # output_dict["loss"] += 0.01* torch.nn.functional.cosine_similarity(self.origin_perturb_emb,  embedded_tokens[0, self.perturb_idx,:], 0, 1e-8)

        return output_dict


        
        
    # def prepare_for_attack(self, origin_perturb_emb=None, perturb_idx=None, language_model=None):
    #     self.attack = True
    #     self.origin_perturb_emb = origin_perturb_emb
    #     self.perturb_idx = perturb_idx
    #     self.language_model = language_model

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
