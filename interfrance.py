from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer = pipeline("summarization", model="Mahmoud3899/bart-multilingual-final", tokenizer="Mahmoud3899/bart-multilingual-final",device=device)


def get_summary(article):
    summary = summarizer(article)
    summary = summary[0]["summary_text"]
    return summary