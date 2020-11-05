"""

IDE: PyCharm
Project: pdf-explorer
Author: Robin
Filename: tokenization.py
Date: 05.11.2020

"""
import os

import streamlit as st
import textract
from bpemb import BPEmb


@st.cache
def extract_text(file_path):
    if os.path.exists(file_path):
        text = textract.process(file_path)
        text_utf8 = text.decode("utf-8")
        return text_utf8
    else:
        return None


@st.cache
def detokenize(bpemb: BPEmb, subwords: []):
    return bpemb.decode(subwords)


@st.cache
def tokenize_bpemb():
    pass
