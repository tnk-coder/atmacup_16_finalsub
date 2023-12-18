# binary
"""
lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'verbose': -1,
              'n_jobs': 8, 'seed': CFG.seed, 'learning_rate': 0.1,
              'metric': 'auc',
              'num_leaves': 64,
              'max_depth': 5,
              'bagging_seed': CFG.seed,
              'feature_fraction_seed': CFG.seed,
              'drop_seed': CFG.seed,
              }

"""
# regression
"""
lgb_params = {'objective': 'regression', 'boosting_type': 'gbdt', 'verbose': -1,
              'n_jobs': 8, 'seed': CFG.seed, 'learning_rate': 0.1, 'n_estimators': 3000,
              'metric': 'rmse'
              }
"""

# multiclass
"""
lgb_params = {'objective': 'multiclass', 'boosting_type': 'gbdt', 'verbose': -1,
            'n_jobs': 8, 'seed': CFG.seed, 'learning_rate': 0.1,
            'num_class': CFG.num_class, # multiclassなら必要
            'num_leaves': 64,
            'max_depth': 5,
            'bagging_seed': CFG.seed,
            'feature_fraction_seed': CFG.seed,
            'drop_seed': CFG.seed,
            }
"""

# nfl3 pp
"""
lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'verbose': -1,
              'n_jobs': 8, 'seed': CFG.seed, 'learning_rate': 0.1,
              'metric': 'auc',
              'num_leaves': 64,
              'max_depth': 5,
              'bagging_seed': CFG.seed,
              'feature_fraction_seed': CFG.seed,
              'drop_seed': CFG.seed,
              }
"""

# fb4
"""
lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'verbose': -1,
              'n_jobs': 8, 'seed': CFG.seed, 'learning_rate': 0.1,
              'metric': 'auc',
              'num_leaves': 64,
              'max_depth': 5,
              'bagging_seed': CFG.seed,
              'feature_fraction_seed': CFG.seed,
              'drop_seed': CFG.seed,
              }
"""

# atmacup15 lgb
"""
lgb_params = {'objective': CFG.objective_cv, 'boosting_type': 'gbdt', 'verbose': -1,
            'n_jobs': 8, 'seed': CFG.seed, 'learning_rate': 0.1,
            # 'num_class': CFG.num_class, # multiclassなら必要
            'num_leaves': 64,
            'metric': 'rmse',
            'max_depth': 5,
            'bagging_seed': CFG.seed,
            'feature_fraction_seed': CFG.seed,
            'drop_seed': CFG.seed,
            }
"""

# atmacup 15 catboost
"""
lgb_params = {
            'random_seed': CFG.seed,'learning_rate': lr, 'iterations': 10000,
            'loss_function': 'RMSE', 'task_type': 'GPU',
            }
"""
