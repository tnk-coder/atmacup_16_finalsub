from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import List

class AbstractBaseBlock:
    """
    https://www.guruguru.science/competitions/16/discussions/95b7f8ec-a741-444f-933a-94c33b9e66be/
    """

    def __init__(self) -> None:
        pass

    def fit(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        # return self.transform(input_df)
        raise NotImplementedError()

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


def run_block(input_df: pd.DataFrame, blocks: List[AbstractBaseBlock], is_fit):
    output_df = pd.DataFrame()
    for block in blocks:
        name = block.__class__.__name__

        if is_fit:
            print(f'fit: {name}')
            _df = block.fit(input_df)
        else:
            print(f'transform: {name}')
            _df = block.transform(input_df)

        print(f'concat: {name}')
        output_df = pd.concat([output_df, _df], axis=1)
    return output_df

class NumericBlock(AbstractBaseBlock):
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = input_df[self.col].copy()
        return output_df

class LabelEncodingBlock(AbstractBaseBlock):
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col
        self.encoder = LabelEncoder()

    def fit(self, input_df):
        # return self.transform(input_df)

        self.encoder.fit(input_df[self.col])
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()

        # output_df[self.col] = self.encoder.fit_transform(input_df[self.col])

        # self.encoder.fit(input_df[self.col])
        output_df[self.col] = self.encoder.transform(input_df[self.col])
        return output_df.add_suffix('@le')

class CountEncodingBlock(AbstractBaseBlock):
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col

    def fit(self, input_df):
        self.val_count_dict = {}
        self.val_count = input_df[self.col].value_counts()
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = input_df[self.col].map(self.val_count)
        return output_df.add_suffix('@ce')

class AggBlock(AbstractBaseBlock):
    def __init__(self, grp_col: str, target_cols: List[str], agg_cols: List[str]) -> None:
        super().__init__()
        self.grp_col = grp_col
        self.target_cols = target_cols
        self.agg_cols = agg_cols

    def fit(self, input_df):
        self.grp_df = input_df.groupby(self.grp_col)[
            self.target_cols].agg(self.agg_cols)
        self.grp_df.columns = [f'{self.grp_col}_' +
                               '_'.join(c) for c in self.grp_df.columns]
        # self.grp_df.add_prefix(f'{self.grp_col}_')
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.grp_col] = input_df[self.grp_col]
        output_df = output_df.merge(self.grp_df, on=self.grp_col, how='left')
        output_df.drop(self.grp_col, axis=1, inplace=True)
        return output_df

class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self, col: str, target_col: str) -> None:
        super().__init__()
        self.col = col
        self.target_col = target_col

    def fit(self, input_df):
        output_df = pd.DataFrame(index=input_df.index.values)
        folds = input_df['fold']
        for fold in input_df['fold'].unique():
            idx_train, idx_valid = (folds != fold), (folds == fold)
            group = input_df[idx_train].groupby(
                self.col)[self.target_col].mean().to_dict()
            output_df.loc[idx_valid,
                          f'{self.col}@te'] = input_df.loc[idx_valid, self.col].map(group)
        self.group = input_df.groupby(
            self.col)[self.target_col].mean().to_dict()
        return output_df

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[f'{self.col}@te'] = input_df[self.col].map(self.group)
        return output_df
