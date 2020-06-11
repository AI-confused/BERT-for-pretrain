from sklearn.metrics import f1_score
import argparse
import csv
import logging
import os
import random
import unicodedata
import sys
from io import open
import pandas as pd
import numpy as np
import torch
import time
import collections
import torch.nn as nn
from collections import defaultdict
import gc
from tqdm import tqdm
import itertools
from multiprocessing import Pool
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
import functools

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class Tokenizer:
    """This class provide a basic tokenization for all pre-trained models.
    Examples:
        Unicode: https://unicode-table.com/cn
            kwargs["pad_token"] = chr(0x0020) # whitespace
            kwargs["cls_token"] = chr(0x0000)
            kwargs["sep_token"] = chr(0x001F)
            kwargs["mask_token"] = chr(0x001E)
            kwargs["unk_token"] = chr(0x001D)
    """
    def __init__(
            self,
            tokens_folder: str
    ):
        """Init.
        Parameters:
            tokens_folder (str):
                The root folder where the vocabulary `vocab.txt` exists.
        """
        super(Tokenizer, self).__init__()
        self._tokens_folder = tokens_folder
        self._to_ids = {}
        self._to_token = []
        self._load_tokens()
        
    def save(
            self,
            output_folder: str
    ):
        """Save vocab to somewhere.
        Parameters:
            output_folder (str):
                The folder where vocab to be saved.
        Returns (None)
        """
        file_name = os.path.join(output_folder, "vocab.txt")
        with open(file_name, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(self._to_token))
            
    def _load_tokens(
            self
    ):
        """Load all vocabs from disk.
        """
        file_name = os.path.join(self._tokens_folder, "vocab.txt")
        with open(file_name, "r", encoding="utf-8") as f_in:
            for line in f_in.readlines():
                token = line.replace("\n", "")
                if token != "":
                    self._to_ids[token] = len(self)
                    self._to_token.append(token)
        logger.info('loading tokenizer from {}'.format(file_name))
        
    def __len__(
            self
    ):
        """Get the total number of vocabs.
        """
        return len(self._to_token)
    def __getitem__(
            self,
            id_or_item
    ):
        """Get mapped item from index or token.
        Parameters:
            id_or_item (str, int):
                Convert token into index or inde into token.
        Returns (int, str):
            Index or Token.
        """
        if isinstance(id_or_item, int) \
                and id_or_item >= 0 \
                and id_or_item < len(self):
            return self._to_token[id_or_item]
        if isinstance(id_or_item, str) and id_or_item in self._to_ids.keys():
            return self._to_ids[id_or_item]
        return self._to_ids[self.unk_token]
    @property
    def special_tokens(
            self
    ):
        """Return all special tokens.
        """
        return ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
    @property
    def sep_token(
            self
    ):
        """Separation token.
        """
        return "[SEP]"
    @property
    def sep_id(
            self
    ):
        """Separation token id.
        """
        return self[self.sep_token]
    @property
    def pad_token(
            self
    ):
        """Pad token.
        """
        return "[PAD]"
    @property
    def pad_id(
            self
    ):
        """Pad token id.
        """
        return self[self.pad_token]
    @property
    def cls_token(
            self
    ):
        """CLS token.
        """
        return "[CLS]"
    @property
    def cls_id(
            self
    ):
        """CLS token id.
        """
        return self[self.cls_token]
    @property
    def mask_token(
            self
    ):
        """MASK token.
        """
        return "[MASK]"
    @property
    def mask_id(
            self
    ):
        """MASK token id.
        """
        return self[self.mask_token]
    @property
    def unk_token(
            self
    ):
        """UNK token.
        """
        return "[UNK]"
    @property
    def unk_id(
            self
    ):
        """UNK token id.
        """
        return self[self.unk_token]
    @property
    def empty_token(
            self
    ):
        """EMPTY token.
        """
        return "[EMPTY]"
    @property
    def empty_id(
            self
    ):
        """EMPTY token id.
        """
        return self[self.empty_token]
    def convert_tokens_to_ids(
            self,
            tokens: list
    ):
        """Convert a token list into a index list.
        Parameters:
            tokens (list):
                A token list.
        Returns (list):
            A index list.
        """
        return [self[t] for t in tokens]
    def tokenize(
            self,
            text: str
    ):
        """Tokenize a string.
        Parameters:
            text (str): A string.
        Returns (list):
            Tokens.
        """
        raise NotImplementedError
    def random_token(
            self
    ):
        """Get a random token without special token.
        Returns (str):
            A random token.
        """
        spe = self.special_tokens
        token = self[random.randint(0, len(self) - 1)]
        count = 100
        while token in spe and count >= 0:
            token = self[random.randint(0, len(self) - 1)]
            count -= 1
        if token in spe:
            print("Token Random Choice Failed.", token)
        return token
    

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 output_ids,

    ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.output_ids = output_ids
        
        
def load_and_cache_examples(args, tokenizer, is_training):
    # Load data features from cache or dataset file
    if is_training==1:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
#                 str(args.train_language),
                ),
        )   
    elif is_training==2:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                "dev",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
#                 str(args.train_language),
                ),
        )
    else:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                "predict",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
#                 str(args.train_language),
                ),
        )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if is_training==1:
            examples = read_examples(os.path.join(args.data_dir, 'train.csv'), is_training)
        elif is_training==2:
            examples = read_examples(os.path.join(args.data_dir, 'dev.csv'), is_training)
        else:
            examples = read_examples(os.path.join(args.data_dir, 'test.csv'), is_training)
        features = convert_examples_to_features(
            examples, tokenizer, args.max_seq_length, is_training)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_output_ids = torch.tensor([f.output_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_output_ids)
    return dataset
        
        
def read_examples(input_file, is_training):
    df=pd.read_csv(input_file)
    if is_training==1 or is_training==2:
        examples=[]
        for val in tqdm(df[['id','text']].values, desc="read train or dev examples"):
            examples.append(InputExample(guid=val[0],text_a=val[1]))
    else:
        examples=[]
        for val in tqdm(df[['id', 'text']].values, desc="read test examples"):
            examples.append(InputExample(guid=val[0],text_a=val[1]))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for example_index, example in tqdm(enumerate(examples), total=len(examples), desc="convert squad examples to features"):
        # mask lm preprocess
        text = example.text_a.replace("\n", "")
        tokens = list(text)
        _truncate_seq_pair(tokens, '', max_seq_length-2)
        masks = [1] # cls
        output_ids = tokenizer.convert_tokens_to_ids(tokens)
        for i in range(0, len(tokens)):
            prob1 = random.random()
            if prob1 < 0.15:
                masks.append(0)
                prob2 = random.random()
                if prob2 < 0.8:
                    tokens[i] = tokenizer.mask_token
                elif prob2 >= 0.8 and prob2 < 0.9:
                    tokens[i] = tokenizer.random_token()
            else:
                output_ids[i] = -100
                masks.append(1)
        masks.append(1) # sep
        output_ids = [-100] + output_ids + [-100]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(tokens)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        output_ids += [-100] * padding_length
        masks += ([0] * padding_length)
        segment_ids += ([0] * padding_length)

        if example_index == 1 and is_training==1:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581','_')))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("output_ids: {}".format(' '.join(map(str, output_ids))))
            logger.info("input_mask: {}".format(' '.join(map(str, masks))))
            logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))

        features.append(
            InputFeatures(
                example_id=example.guid,
                input_ids = input_ids,
                input_mask = masks,
                segment_ids = segment_ids,
                output_ids = output_ids,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()
        
def get_f1(preds, labels):
    return f1_score(labels, preds, labels=[0,1], average='macro')
    
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
