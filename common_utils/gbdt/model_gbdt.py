import lightgbm as lgb
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
import pandas as pd
import matplotlib.pyplot as plt
import pickle
'''
def get_model(cat_cols, text_cols):
    """
    # binary
    lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'verbose': -1,
                  'n_jobs': 8, 'seed': CFG.seed, 'learning_rate': 0.01,
                  'num_class': CFG.target_size,
                  'num_leaves': 64,
                  'max_depth': 5,
                  'bagging_seed': CFG.seed,
                  'feature_fraction_seed': CFG.seed,
                  'drop_seed': CFG.seed
                  }
    """


    model = LightGBM(lgb_params=lgb_params,
                     imp_dir=CFG.figures_dir, save_dir=CFG.model_dir)
    return model


def get_fit_params():
    # params = None
    # lgb
    params = {
        'num_boost_round': 100000,
        'early_stopping_rounds': 50,
        'verbose_eval': 50
    }

    return params
'''


class LightGBM:

    def __init__(self, lgb_params, save_dir=None, imp_dir=None, categorical_feature=None,
                 model_name='lgb',
                 stopping_rounds=50) -> None:
        self.save_dir = save_dir
        self.imp_dir = imp_dir
        self.lgb_params = lgb_params
        self.categorical_feature = categorical_feature

        # saveの切り替え用
        self.model_name = model_name

        self.stopping_rounds = stopping_rounds

    def fit(self, x_train, y_train, **fit_params) -> None:

        X_val, y_val = fit_params['eval_set'][0]
        del fit_params['eval_set']

        train_dataset = lgb.Dataset(
            x_train, y_train, categorical_feature=self.categorical_feature)

        val_dataset = lgb.Dataset(
            X_val, y_val, categorical_feature=self.categorical_feature)

        self.model = lgb.train(params=self.lgb_params,
                               train_set=train_dataset,
                               valid_sets=[train_dataset, val_dataset],
                               callbacks=[lgb.early_stopping(stopping_rounds=self.stopping_rounds,
                                                             verbose=True),
                                          lgb.log_evaluation(50)],
                               **fit_params
                               )

    def plot_importance(self, fold):
        gain_importances = pd.DataFrame({'Feature': self.model.feature_name(),
                                         'Importance': self.model.feature_importance(importance_type='gain')})

        gain_importances.to_csv(
            f'{self.imp_dir}lgb_imp_fold_{fold}_{self.model_name}.csv', index=False)

        gain_importances = gain_importances.nlargest(
            50, 'Importance', keep='first').sort_values(by='Importance', ascending=True)

        gain_importances[['Importance', 'Feature']].plot(
            kind='barh', x='Feature', color='blue', figsize=(12, 8), fontsize=9)
        # plt.ylabel('Feature', fontsize=12)
        plt.title(f'gain importance fold {fold}')
        plt.savefig(
            f'{self.imp_dir}gain_importance_fold{fold}_{self.model_name}.png', bbox_inches='tight')

    def plot_importance_all(self, n_fold=5):
        dfs = [pd.read_csv(
            f'{self.imp_dir}lgb_imp_fold_{fold}_{self.model_name}.csv') for fold in range(n_fold)]
        imp_df = pd.concat(dfs).reset_index(drop=True)
        imp_df = imp_df.groupby(['Feature'])['Importance'].mean().reset_index()

        gain_importances = imp_df.nlargest(
            50, 'Importance', keep='first').sort_values(by='Importance', ascending=True)
        gain_importances[['Importance', 'Feature']].plot(
            kind='barh', x='Feature', color='blue', figsize=(12, 8), fontsize=9)
        plt.title('gain importance all')
        plt.savefig(
            f'{self.imp_dir}gain_importance_all_{self.model_name}.png', bbox_inches='tight')

        return imp_df

    def save(self, fold):
        save_to = f'{self.save_dir}lgb_fold_{fold}_{self.model_name}.txt'
        self.model.save_model(save_to)

    def load(self, fold):
        load_from = f'{self.save_dir}lgb_fold_{fold}_{self.model_name}.txt'
        self.model = lgb.Booster(model_file=load_from)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class CatBoost:

    def __init__(self, lgb_params, mode, save_dir=None, imp_dir=None, categorical_feature=None, text_features=None,
                 model_name='catboost') -> None:
        self.mode = mode
        self.save_dir = save_dir
        self.imp_dir = imp_dir
        self.lgb_params = lgb_params
        self.categorical_feature = categorical_feature
        self.text_features = text_features

        # saveの切り替え用
        self.model_name = model_name

    def fit(self, x_train, y_train, **fit_params) -> None:

        print('categorical feature')
        print(self.categorical_feature)

        print('text feature')
        print(self.text_features)

        X_val, y_val = fit_params['eval_set'][0]
        del fit_params['eval_set']

        train_pool = Pool(x_train, y_train, text_features=self.text_features,
                          cat_features=self.categorical_feature)
        val_pool = Pool(X_val, y_val, text_features=self.text_features,
                        cat_features=self.categorical_feature)

        if self.mode == 'regression':
            self.model = CatBoostRegressor(**self.lgb_params)
        else:
            self.model = CatBoostClassifier(**self.lgb_params)

        self.model.fit(train_pool,
                       eval_set=val_pool,
                       **fit_params
                       )

    def plot_importance(self, fold):
        gain_importances = pd.DataFrame({'Feature': self.model.feature_names_,
                                         'Importance': self.model.feature_importances_})

        gain_importances.to_csv(
            f'{self.imp_dir}catboost_imp_fold_{fold}_{self.model_name}.csv', index=False)

        gain_importances = gain_importances.nlargest(
            50, 'Importance', keep='first').sort_values(by='Importance', ascending=True)
        gain_importances[['Importance', 'Feature']].plot(
            kind='barh', x='Feature', color='blue', figsize=(12, 8), fontsize=9)

        # plt.ylabel('Feature', fontsize=12)
        plt.title(f'gain importance fold {fold}')
        plt.savefig(
            f'{self.imp_dir}gain_importance_fold{fold}_{self.model_name}.png', bbox_inches='tight')

    def plot_importance_all(self, n_fold=5):
        dfs = [pd.read_csv(
            f'{self.imp_dir}catboost_imp_fold_{fold}_{self.model_name}.csv') for fold in range(n_fold)]
        imp_df = pd.concat(dfs).reset_index(drop=True)
        imp_df = imp_df.groupby(['Feature'])['Importance'].mean().reset_index()

        gain_importances = imp_df.nlargest(
            50, 'Importance', keep='first').sort_values(by='Importance', ascending=True)
        gain_importances[['Importance', 'Feature']].plot(
            kind='barh', x='Feature', color='blue', figsize=(12, 8), fontsize=9)
        plt.title('gain importance all')
        plt.savefig(
            f'{self.imp_dir}gain_importance_all_{self.model_name}.png', bbox_inches='tight')

        return imp_df

    def save(self, fold):
        save_to = f'{self.save_dir}catboost_fold_{fold}_{self.model_name}.pkl'
        pickle.dump(self.model, open(save_to, 'wb'))

        # debug
        # self.model = pickle.load(open(save_to, 'rb'))

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)
        # return self.model.predict(x, prediction_type='Probability')[:, 1]
