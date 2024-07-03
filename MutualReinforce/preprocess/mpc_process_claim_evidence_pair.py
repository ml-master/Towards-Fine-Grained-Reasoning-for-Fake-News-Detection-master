'''
Get data related to each news
'''
import argparse
import multiprocessing as mp
import os
import pickle
import warnings
from functools import partial
from multiprocessing import Pool, Manager

import pandas as pd
from nltk.tokenize import sent_tokenize

import config

import sys
sys.path.append("..")
from utils import utils

DEBUG = False

warnings.filterwarnings("always")


def process_example(filename, dataset_name, args, global_news_article_d, news_article_df, examples):
    '''
    If evidence exists, use ent as title, original sentence as claim, summary as evidence
    If no evidence, use the original text

    :param filename:
    :param dataset_name:
    :param args:
    :param global_news_article_d:
    :param news_article_df:
    :param examples: result dict
    :var sent_li: a list of sentences in original news articld
    :var evi_li: a list of sublists. each sublist contains a claim (related sentence from the news article), a title (the entity or keyword), and the evidence (from Wiki page)
    :return:
    '''
    print(filename)
    example_fullname = f"{config.FAKE_NEWS_DATA}\\{dataset_name}\\{filename}"

    if not filename in global_news_article_d or not filename in news_article_df.index:
        print(f"\t{filename} NOT IN news article dict")
        return

    sent_li = sent_tokenize(global_news_article_d[filename])
    evi_li = []

    if os.path.exists(f"{example_fullname}\\evidence.tsv"):
        evidence_df = pd.read_csv(f"{example_fullname}\\evidence.tsv", sep='\t')

        for i, sent_ent_summary in evidence_df.iterrows():
            ent = sent_ent_summary.loc['ent']
            summary = sent_ent_summary.loc['summary']
            sent = sent_ent_summary.loc['sent']
            evi_li += [[sent, ent, summary]]

    elif os.path.exists(f"{example_fullname}\\evidence_SKIP.txt"):
        print(f"\t{filename} DEFAULT SKIPPED")

    # if len(evi_li) < args.min_evi_list_len:
    #     evi_li += sent_li
    label = news_article_df.loc[filename].label
    examples[filename] = (evi_li, sent_li, label)


def get_entities_in_text(filename, nlp, global_news_article_d):
    if not filename in global_news_article_d:
        return
    try:
        doc = nlp(global_news_article_d[filename])
        ents_keep = {'PERSON', 'NORP', 'ORG', 'GPE'}

        ent_d = {}

        for ent in doc.ents:

            # Note: we ignore repeating entities for now
            if ent in ents_keep:
                continue

            if ent.label_ in ents_keep:
                # We can also get the sentence by
                # doc[ent.sent.start:ent.sent.end]
                ent_d[ent.text] = {
                    "label": ent.label_,
                    "start": ent.start,
                    "end": ent.end,
                    "sent": ent.sent.text,
                    "sent_start": ent.sent.start,
                    "sent_end": ent.sent.end
                }

        return ent_d
    except KeyError as e:
        print(f"\t{filename} NOT FOUND in news article df")
        return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--overwrite', action="store_true", help='Whether to overwrite *.tsv')
    parser.add_argument('--add_last_entity', action="store_true",
                        help='Whether to add the actual entity, which we used as the search query on wikipedia')

    parser.add_argument('--min_evi_list_len', type=int, default=10,
                        help='The minimum number of evidence sentences we keep for each example')
    parser.add_argument('--dataset', choices=["politifact", "gossipcop", "both"], default="gossipcop",
                        help='Which dataset to preprocess. ')
    args = parser.parse_args()

    data_list = utils.get_data_list()
    manager = Manager()

    DATASET_NAMES = utils.get_dataset_names(args.dataset)

    # dataset_name is "politifact" or "gossipcop"
    for dataset_name, dataset_list in DATASET_NAMES.items():


        print("#" * 30 + f"\n# Processing {dataset_name}\n" + "#" * 30 + "\n")

        if DEBUG:
            job_list = [["politifact15501", config.POLITIFACT_FAKE_NAME],
                        ['politifact11773', config.POLITIFACT_FAKE_NAME]]
            n_cpu = 1

        else:
            n_cpu = mp.cpu_count()
            job_list = []
            # Either X_real or X_fake
            for dataset in dataset_list:
                for filename in data_list[dataset]:
                    job_list += [[filename, dataset]]

        pool = Pool(processes=n_cpu)
        global_news_article_d = {}
        examples = manager.dict()
        utils.read_news_articles_text(global_news_article_d, dataset_name)
        news_article_df = utils.read_news_articles_labels(dataset_name)
        news_article_df.set_index('id', inplace=True)

        partial_process_example = partial(process_example, args=args,
                                          global_news_article_d=global_news_article_d, news_article_df=news_article_df,
                                          examples=examples)
        pool.starmap(partial_process_example, job_list)
        examples_d = examples._getvalue()
        with open(f"{config.FAKE_NEWS_DATA}\\{dataset_name}_news_article_evidence.pkl", 'wb') as f:
            pickle.dump(examples_d, f)

        # with open(f"{config.FAKE_NEWS_DATA}\\{name}_news_article_evidence.pkl", 'rb') as f:
        #     new_examples = pickle.load(f)
        # print(new_examples)


if __name__ == '__main__':
    main()
