from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from apex import amp
from sklearn.metrics import matthews_corrcoef, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,
                          XLMConfig, XLMTokenizer,
                          XLNetConfig, XLNetTokenizer,
                          RobertaConfig, RobertaTokenizer,
                          BertForSequenceClassification, XLMForSequenceClassification,
                          RobertaForSequenceClassification, get_linear_schedule_with_warmup, AdamW)
from transformers import XLNetForSequenceClassification

from utils.utils_xlnet import (all_examples_to_features,
                               output_modes, processors, print_results)

def set_seed_everywhere(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def load_and_cache_examples(task, tokenizer, args, logger, evaluate=False):
    processor = processors[task](args)
    output_mode = args.output_mode

    mode = 'dev' if evaluate else 'train'
    cached_features_file = os.path.join(args.data_dir,
                                        f"cached_{mode}_{args.model_name}_{args.max_seq_length}_{task}")

    if os.path.exists(cached_features_file) and not args.reprocess_input_data:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)

        features = all_examples_to_features(examples, label_list, args.max_seq_length, tokenizer,
                                            output_mode,
                                            cls_token_at_end=bool(args.model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(args.model_type in ['roberta']),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def train(train_dataset, model, tokenizer, args, logger):
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = math.ceil(t_total * args.warmup_ratio)
    args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    if args.fp16:
        # Set opt_level as 'O1' for now
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    results_train_all, results_eval_all = [], []

    for epoch in train_iterator:
        torch.cuda.empty_cache()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)

        y_pred_train_all = np.array([], dtype=int)
        y_labels_train_all = np.array([], dtype=int)

        for step, batch in enumerate(epoch_iterator):
            model.train()

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                # XLM don't use segment_ids
                'labels': batch[3]
            }
            outputs = model(**inputs)
            train_loss = outputs.loss
            logits = outputs.logits

            y_pred_train = torch.max(F.softmax(outputs.logits, dim=-1), dim=1)[1].cpu().numpy()
            y_labels_train = inputs['labels'].cpu().numpy()

            y_pred_train_all = np.append(y_pred_train_all, y_pred_train)
            y_labels_train_all = np.append(y_labels_train_all, y_labels_train)

            if args.gradient_accumulation_steps > 1:
                train_loss = train_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)

            else:
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += train_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics

                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

        # End of epoch. Evaluate training results
        results, _ = compute_metrics(args.task_name, y_pred_train_all, y_labels_train_all)
        results_train_all += [results]

        print_results(results, epoch, dataset_split_name="Train", args=args)

        # Evaluate at the end of epoch
        # Only evaluate when single GPU otherwise metrics may not average well
        if args.evaluate_every_nepochs > 0 and (epoch+1) % args.evaluate_every_nepochs == 0:
            results, _ = evaluate(model, tokenizer, args=args, logger=logger, prefix=str(epoch))
            print_results(results, epoch, dataset_split_name="Eval", args=args)
            results_eval_all += [results]

            for key, value in results.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

    results_train_all_df = pd.DataFrame.from_dict(results_train_all)
    results_eval_all_df  = pd.DataFrame.from_dict(results_eval_all)

    results_train_all_df.to_csv(f"{args.output_dir}/{args.model_type}_{args.max_seq_length}_Train.tsv", sep='\t')
    results_eval_all_df.to_csv(f"{args.output_dir}/{args.model_type}_{args.max_seq_length}_Eval.tsv", sep='\t')

    return global_step, tr_loss / global_step


def get_eval_report(labels, preds, processor=None):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    def get_mismatched(labels, preds, data_dir='data'):
        mismatched = labels != preds
        examples = processor.get_dev_examples(data_dir)
        wrong = [i for (i, v) in zip(examples, mismatched) if v]
        return wrong

    results = {
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pre": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='micro')
    }

    if not processor:
        return results, None

    return results, get_mismatched(labels, preds)


def compute_metrics(task_name, preds, labels, processor=None):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds, processor)


def evaluate(model, tokenizer, args, logger, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    results = {}
    EVAL_TASK = args.task_name
    processor = processors[EVAL_TASK](args)
    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True, args=args, logger=logger)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss_all = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                # XLM don't use segment_ids
                'labels': batch[3]
            }
            outputs = model(**inputs)
            eval_loss, logits = outputs[:2]

            eval_loss_all += eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss_all = eval_loss_all / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    result, wrong = compute_metrics(EVAL_TASK, preds, out_label_ids, processor)
    results.update(result)

    return results, wrong


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Transformers Sequence Classification')

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset_name', type=str, default='politifact')
    parser.add_argument('--model_type', type=str, default='xlnet')
    parser.add_argument('--model_name', type=str, default='xlnet-base-cased')
    parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--cache_dir', type=str, default='cache/')
    parser.add_argument('--task_name', type=str, default='binary')
    parser.add_argument('--sorted_by_timestamp', action='store_true', default=True)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_eval', action='store_true', default=True)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--output_mode', type=str, default='classification')
    parser.add_argument('--seed', type=int, default=21)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=4e-5)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--test_size', type=float, default=0.2)

    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)

    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--evaluate_every_nepochs', type=int, default=1, help="How many epochs to evaluate on dev set. Set to 0 if no evaluation")
    parser.add_argument('--eval_all_checkpoints', action='store_true', default=False)
    parser.add_argument('--reprocess_input_data', action='store_true', default=False,
                        help="Whether to reprocess the cached training and testing features. Default set to False.")
    parser.add_argument('--overwrite_output_dir', action='store_true', default=False)


    args = parser.parse_args()

    set_seed_everywhere(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overpcome.".format(
                args.output_dir))

    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }

    print(f"Using {args.model_type}, {args.model_name}")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_name, num_labels=2, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

    model = model_class.from_pretrained(args.model_name)

    model.to(args.device)

    task = args.task_name

    if task in processors.keys() and task in output_modes.keys():
        processor = processors[task](args)
        label_list = processor.get_labels()
    else:
        raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')

    if args.do_train:
        train_dataset = load_and_cache_examples(task, tokenizer, args=args, logger=logger)
        global_step, tr_loss = train(train_dataset, model, tokenizer, args=args, logger=logger)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)

        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, wrong_preds = evaluate(model, tokenizer, args=args, logger=logger, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)


if __name__ == "__main__":
    main()
