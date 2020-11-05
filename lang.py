"""

IDE: PyCharm
Project: pdf-explorer
Author: Robin
Filename: lang.py
Date: 05.11.2020

"""
import spacy
from bpemb import BPEmb


def get_lang_code(language: str):
    if language == "German":
        return "de"
    elif language == "English":
        return "en"
    else:
        raise Exception("Unknown languuage: %s" % language)


def get_models_for_language(language: str, token_vector_dims=50):
    lang_code = get_lang_code(language)

    spacy_model_name = None
    if lang_code == "de":
        spacy_model_name = "de_core_news_sm"

    elif lang_code == "en":
        spacy_model_name = "en_core_web_sm"

    if spacy_model_name is None:
        raise Exception("Language not supported: %s" % lang_code)

    nlp_model = spacy.load(spacy_model_name)
    tokenizer_model = BPEmb(lang=lang_code, dim=token_vector_dims)

    return lang_code, nlp_model, tokenizer_model
