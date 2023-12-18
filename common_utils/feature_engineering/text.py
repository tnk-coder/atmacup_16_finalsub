import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from typing import List

from .base import AbstractBaseBlock

class TfidfSVDBlock(AbstractBaseBlock):
    def __init__(self, col: str, n_components: int = 128, max_features: int = 100000) -> None:
        super().__init__()
        self.col = col
        self.pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=max_features)),
                ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ]
        )

    def fit(self, input_df):
        vec = self.pipe.fit_transform(input_df[self.col])
        output_df = pd.DataFrame(vec).add_prefix(f"{self.col}@Tfidf_SVD_")
        return output_df

    def transform(self, input_df):
        vec = self.pipe.transform(input_df[self.col])
        output_df = pd.DataFrame(vec).add_prefix(f"{self.col}@Tfidf_SVD_")
        return output_df

class TextLenBlock(AbstractBaseBlock):
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col + '@len'] = input_df[self.col].str.len()
        output_df[self.col +
                  '@num_word'] = input_df[self.col].str.split().str.len()
        return output_df
