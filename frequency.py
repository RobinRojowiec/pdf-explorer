"""

IDE: PyCharm
Project: pdf-explorer
Author: Robin
Filename: frequency.py
Date: 05.11.2020

"""
import pandas as pd
import streamlit as st
from wordfreq import zipf_frequency


@st.cache
def get_freq_df(word_counter, pos_tag_dict, top_k_words, lang_code):
    # sort and cut by frequency of each word in document
    sorted_list_doc_freqs = sorted(word_counter.items(), key=lambda x: x[1])
    sorted_list_doc_freqs = sorted_list_doc_freqs[:top_k_words]

    # # get zipf frequency
    # sorted_list_zipf_freqs = []
    # for sorted_pair in sorted_list_doc_freqs:
    #     zipf_freq = zipf_frequency(sorted_pair[0], lang_code)  # log of relative frequency
    #     sorted_list_zipf_freqs.append([sorted_pair[0], zipf_freq])
    # sorted_list_zipf_freqs = sorted(sorted_list_zipf_freqs, key=lambda x: x[1])
    #
    # # combine rank global and document
    # combined_ranks = defaultdict(int)
    # for i in range(len(sorted_list_doc_freqs)):
    #     combined_ranks[sorted_list_doc_freqs[i][0]] += i + 1
    #
    # for i in range(len(sorted_list_zipf_freqs)):
    #     combined_ranks[sorted_list_zipf_freqs[i][0]] += i + 1

    word_df = pd.DataFrame()
    for word_freq_pair in sorted_list_doc_freqs:
        word = word_freq_pair[0]
        word_df = word_df.append({"Word": word, "Doc-Freq": word_freq_pair[1],
                                  "Global Freq": zipf_frequency(word, lang_code), "POS-Tag": pos_tag_dict[word]},
                                 ignore_index=True)
    return word_df
