# atmacup 16 4th place solution

# 解法
https://www.guruguru.science/competitions/22/discussions/f1c6b804-6c60-4987-8a6a-572052dc20a5/

# 実行環境
Ubuntu 20.04 LTS
GPU: RTX 3090
memory: 64GB
Python 3.10.9
# 準備
- datasets/atmaCup16_Dataset 内にデータを置く

# 候補生成、embedding特徴量の作成
- candidate_generation内のnotebookを全て実行する

# フィルタリング
- gbdt/train_gbdt/train_gbdt_1stage_feature_only_exp001.ipynb を実行して、特徴量生成
- gbdt/train_gbdt/train_gbdt_1stage_training_exp002.ipynb を実行して、候補をtop40に絞る

# Reranker
- gbdt/train_gbdt/train_gbdt_2stage_feature_only_sep_train_test_exp002.ipynb を実行して、特徴量生成
- gbdt/train_gbdt/train_gbdt_2stage_training_sep_train_test_exp002.ipynb を実行して、Rankerを学習、最終subを作成
  - 最終subのcsvは outputs/atmacup_16/train_gbdt_2stage_training_sep_train_test_exp002/submissions フォルダ内に作成される