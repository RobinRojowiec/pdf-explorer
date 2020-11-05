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
import pandas as pd
import streamlit as st
from matplotlib.colors import to_hex

# prepare dir
from clustering import do_clustering
from frequency import get_freq_df
from lang import get_models_for_language
from tokenization import extract_text

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

lang_code, nlp, tokenizer = get_models_for_language(option)
analyzed = None

uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'])
if uploaded_file is not None:
    filename = uploaded_file.name
    bytes_data = uploaded_file.read()
    file_path = tmp_dir + filename

    if len(bytes_data) > 0:
        with open(tmp_dir + filename, "wb") as out_file:
            out_file.write(bytes_data)

    text_utf8 = extract_text(file_path)
    if text_utf8 is not None:
        my_bar = st.progress(1)

        # tokenize and analyze using spacy
        analyzed = nlp(text_utf8)

        st.subheader("Entities")
        ent_df = pd.DataFrame(columns=["Entity", "Label"])
        for ent in analyzed.ents:
            ent_df = ent_df.append({"Entity": ent.text, "Label": ent.label_}, ignore_index=True)
        st.dataframe(ent_df)

        st.header("Words")
        interesting_pos_tags = st.multiselect("Filter by POS-Tag",
                                              [
                                                  "ADP",
                                                  "ADV",
                                                  "AUX",
                                                  "CONJ",
                                                  "CCONJ",
                                                  "DET",
                                                  "INTJ",
                                                  "NOUN",
                                                  "NUM",
                                                  "PART",
                                                  "PRON",
                                                  "PROPN",
                                                  "PUNCT",
                                                  "SCONJ",
                                                  "SYM",
                                                  "VERB",
                                                  "X",
                                                  "SPACE"], default=[]
                                              )

        max_len = len(analyzed)
        token_counter = 0

        words = []
        vectors = []
        word_counter = defaultdict(int)
        pos_tag_dict = dict()

        for token in analyzed:
            lemma = token.lemma_
            if (len(interesting_pos_tags) > 0 and token.pos_ in interesting_pos_tags) or len(interesting_pos_tags) == 0:
                if not token.is_stop and token.is_alpha:
                    if lemma not in words:
                        pos_tag_dict[lemma] = token.pos_
                        words.append(lemma)
                        vectors.append(token.vector)

                    # count word occurence
                    word_counter[lemma] += 1

            my_bar.progress(token_counter / max_len)
            token_counter += 1.0 / max_len
        my_bar.progress(1.0)

        st.subheader("Word Frequency")
        top_k_words = 1000
        word_df = get_freq_df(word_counter, pos_tag_dict, top_k_words, lang_code)
        st.dataframe(word_df)

        st.subheader("Clustering Words")
        with st.spinner('Clustering...'):
            eps = st.slider("Maximum Vector-Distance:", 0.01, 10.0, value=1.0)
            min_samples = st.slider("Minimum Samples per Cluster:", 2, 100)

            clusters, num_clusters, noise_counter, cluster_counter, tsne_results, labels = do_clustering(words, vectors,
                                                                                                         eps,
                                                                                                         min_samples)
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
