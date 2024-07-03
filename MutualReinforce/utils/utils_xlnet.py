from __future__ import absolute_import, division, print_function

import csv
import logging
from functools import partial
from multiprocessing import Pool, cpu_count

from transformers.data.processors.utils import DataProcessor

from utils.utils import read_news_articles_labels, read_news_articles_text

logger = logging.getLogger(__name__)
csv.field_size_limit(2147483647)

from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEBUG = 0


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class BinaryProcessor(DataProcessor):
    """Processor for the binary data sets"""

    def __init__(self, args):
        self.news_article_df = read_news_articles_labels(args.dataset_name)
        self.news_article_df.set_index('id', inplace=True)
        if args.sorted_by_timestamp:
            self.news_article_df.sort_values('timestamp', inplace=True, axis=0)

        y = self.news_article_df.label

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.news_article_df, y, stratify=y,
                                                                                test_size=args.test_size)
        self.train_li = self.X_train.reset_index().id.to_list()
        self.test_li = self.X_test.reset_index().id.to_list()

        self.global_news_article_d = {}
        read_news_articles_text(self.global_news_article_d)

    def get_train_examples(self, data_dir):
        return self._create_examples("train")

    def get_dev_examples(self, data_dir):
        return self._create_examples("dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, set_type="train"):

        example_li = self.train_li if set_type == "train" else self.test_li
        examples = []
        for filename in example_li:
            guid = "%s-%s" % (set_type, filename)
            text_a = self.global_news_article_d[filename]
            label = self.news_article_df.loc[filename].label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_example_to_feature(example, tokenizer, max_seq_length=128,
                               output_mode="classification",
                               cls_token_at_end=False, cls_token='[CLS]',
                               sep_token='[SEP]',
                               cls_token_segment_id=1, pad_on_left=False, pad_token=0, sequence_a_segment_id=0,
                               sequence_b_segment_id=1,
                               pad_token_segment_id=0,
                               mask_padding_with_zero=True,
                               sep_token_extra=False):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = int(example.label)
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)


def all_examples_to_features(examples, label_list, max_seq_length,
                             tokenizer, output_mode,
                             cls_token_at_end=False, sep_token_extra=False, pad_on_left=False,
                             cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                             sequence_a_segment_id=0, sequence_b_segment_id=1,
                             cls_token_segment_id=1, pad_token_segment_id=0,
                             mask_padding_with_zero=True,
                             process_count=cpu_count()):
    """
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    if DEBUG:
        process_count = 1

    partial_convert_example_to_feature = partial(convert_example_to_feature,
                                                 max_seq_length=max_seq_length,
                                                 tokenizer=tokenizer, output_mode=output_mode,
                                                 cls_token_at_end=cls_token_at_end, cls_token=cls_token,
                                                 sep_token=sep_token,
                                                 sequence_a_segment_id=sequence_a_segment_id,
                                                 sequence_b_segment_id=sequence_b_segment_id,
                                                 cls_token_segment_id=cls_token_segment_id, pad_on_left=pad_on_left,
                                                 pad_token=pad_token, pad_token_segment_id=pad_token_segment_id,
                                                 mask_padding_with_zero=mask_padding_with_zero,
                                                 sep_token_extra=sep_token_extra)

    pool = Pool(processes=process_count)
    features = tqdm(pool.imap(partial_convert_example_to_feature, examples), total=len(examples))
    features = list(features)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def print_results(results, epoch, args, dataset_split_name="Train", enable_logging=True):
    log_str = f"\n[{dataset_split_name}] Epoch {epoch}\n\tPre: {results['pre']:.3f}, Rec: {results['rec']:.3f}\n\tAcc: {results['acc']:.3f}, F1: {results['f1']:.3f}\n"
    print(log_str)
    for key in ["tp", "tn", "fp", "fn", "mcc"]:
        log_str += f"\t{key} = {str(results[key])}\n"


    if enable_logging:
        f = open(f"{args.output_dir}/{args.model_type}_{dataset_split_name}_{args.max_seq_length}_results.txt", "a+")
        f.write(log_str)





processors = {
    "binary": BinaryProcessor
}

output_modes = {
    "binary": "classification"
}
