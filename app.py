"""

IDE: PyCharm
Project: pdf-explorer
Author: Robin
Filename: app.py
Date: 18.10.2020

"""
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import streamlit as st
import textract
from matplotlib.colors import to_hex
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from wordfreq import zipf_frequency

# prepare dir
tmp_dir = "tmp/"
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

# Add title on the page
st.title("PDF Explorer")


# Clear old files
def clear():
    file_list = [f for f in os.listdir(tmp_dir) if os.path.isfile(tmp_dir + f)]
    for filename in file_list:
        os.remove(tmp_dir + filename)


st.subheader("Upload PDF Document")

# Select language
option = st.selectbox('Select Language', ('German', 'English'))

if option == "German":
    nlp_model = "de_core_news_sm"
    lang_code = "de"
else:
    # English default case
    nlp_model = "en_core_web_sm"
    lang_code = "en"

# load nlp model
nlp = spacy.load(nlp_model)
analyzed = None

uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'])
if uploaded_file is not None:
    filename = uploaded_file.name
    bytes_data = uploaded_file.read()
    file_path = tmp_dir + filename

    if len(bytes_data) > 0:
        with open(tmp_dir + filename, "wb") as out_file:
            out_file.write(bytes_data)

    if os.path.exists(file_path):
        text = textract.process(file_path)
        text_utf8 = text.decode("utf-8")

        my_bar = st.progress(1)
        analyzed = nlp(text_utf8)

        max_len = len(analyzed)
        token_counter = 0

        words = []
        vectors = []
        word_counter = defaultdict(int)
        for token in analyzed:
            lemma = token.lemma_
            if not token.is_stop and token.is_alpha:
                if lemma not in words:
                    words.append(lemma)
                    vectors.append(token.vector)

                # count word occurence
                word_counter[lemma] += 1

            my_bar.progress(token_counter / max_len)
            token_counter += 1.0 / max_len
        my_bar.progress(1.0)

        st.subheader("Word Frequency")

        top_k_words = 1000

        # sort and cut by frequency of each word in document
        sorted_list_doc_freqs = sorted(word_counter.items(), key=lambda x: x[1])
        sorted_list_doc_freqs = sorted_list_doc_freqs[:top_k_words]

        # get zipf frequency
        sorted_list_zipf_freqs = []
        for sorted_pair in sorted_list_doc_freqs:
            zipf_freq = zipf_frequency(sorted_pair[0], lang_code)  # log of relative frequency
            sorted_list_zipf_freqs.append([sorted_pair[0], zipf_freq])
        sorted_list_zipf_freqs = sorted(sorted_list_zipf_freqs, key=lambda x: x[1])

        # combine rank global and document
        combined_ranks = defaultdict(int)
        for i in range(len(sorted_list_doc_freqs)):
            combined_ranks[sorted_list_doc_freqs[i][0]] += i + 1

        for i in range(len(sorted_list_zipf_freqs)):
            combined_ranks[sorted_list_zipf_freqs[i][0]] += i + 1

        word_df = pd.DataFrame()
        for word_freq_pair in sorted_list_doc_freqs:
            word = word_freq_pair[0]
            word_df = word_df.append({"Word": word, "Doc-Freq": word_freq_pair[1],
                                      "Global Freq": zipf_frequency(word, lang_code),
                                      "Combined Ranks": combined_ranks[word]}, ignore_index=True)

        st.dataframe(word_df)

        st.subheader("Clustering Words")

        with st.spinner('Clustering...'):
            eps = st.slider("Maximum Vector-Distance:", 0.01, 10.0, value=1.0)
            min_samples = st.slider("Minimum Samples per Cluster:", 2, 100)

            X = np.array(vectors)

            # visualize clusters
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0)
            tsne_results = tsne.fit_transform(X)

            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(tsne_results)

            noise_counter, cluster_counter = 0, 0
            labels = clustering.labels_
            clusters = [[] for label in labels]
            num_clusters = set()
            for i in range(len(labels)):
                word_id = i
                label_id = labels[i]
                # ignore noise = -1
                if label_id >= 0:
                    clusters[label_id].append(words[word_id])
                    cluster_counter += 1
                    num_clusters.add(label_id)
                else:
                    noise_counter += 1
            st.text(
                "Words: " + str(len(words)) + ", Clustered Words: " + str(cluster_counter) + ", Noise Samples: " + str(
                    noise_counter) + ", Clusters: " + str(len(num_clusters)))

            cmap = matplotlib.cm.get_cmap('viridis')
            fig = plt.figure(figsize=(16, 10))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=list(labels), vmin=0, vmax=len(num_clusters),
                        cmap=cmap)
            st.pyplot(fig)

            st.header("Cluster samples")

            # get colors
            norm = matplotlib.colors.Normalize(vmin=0.0, vmax=len(num_clusters))

            cluster_count = 1
            for cluster_id in range(len(clusters)):
                if len(clusters[cluster_id]) > 0:
                    color = to_hex(cmap(norm(cluster_count)))

                    st.markdown(("<h3 style='color:%s'>Cluster " + str(cluster_count) + "</h3>") % color,
                                unsafe_allow_html=True)
                    st.markdown(",".join([" " + word for word in clusters[cluster_id]]))
                    cluster_count += 1

            st.success("Finished!")
