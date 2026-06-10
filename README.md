# 基于 DeepFM 的电影个性化推荐系统

本项目基于 PyTorch 实现 DeepFM 推荐模型，使用 MovieLens 1M 数据集完成电影推荐任务。项目支持数据预处理、模型训练、AUC 指标验证、命令行 Top-N 推荐，以及本地可交互网页演示。

## 项目结构

```text
.
├── main.py                  # 训练、评估、推荐入口
├── recommend_ui.py          # 本地交互式推荐网页服务
├── model/DeepFM.py          # DeepFM 模型定义
├── data/movielens.py        # MovieLens 数据处理与加载
├── data/ml-1m/ml-1m/        # MovieLens 1M 原始数据
├── data/processed/          # 预处理后的训练集、验证集和元数据
├── web/                     # 推荐演示网页
├── ml1m_dual_target_try5.pt # 已训练模型权重
└── ml1m_dual_target_try5.csv# 训练过程记录
```

## 环境依赖

主要依赖如下：

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

若已经安装过 `torch`，只需要确保 `numpy`、`pandas`、`scikit-learn` 等基础库可用即可。

## 模型训练

使用 MovieLens 1M 数据集重新训练模型：

```bash
python main.py ml_train --ml1m_dir .\data\ml-1m\ml-1m --task_type implicit --neg_ratio 1 --max_rows 500000 --split_strategy random --epochs 20 --patience 4 --batch_size 8192 --eval_batch_size 16384 --embedding_size 24 --hidden_dims 256,128 --dropout 0.2,0.2 --lr 0.001 --model_path ml1m_dual_target_try5.pt --history_path ml1m_dual_target_try5.csv
```

训练完成后会生成：

```text
ml1m_dual_target_try5.pt
ml1m_dual_target_try5.csv
```

其中 `.pt` 文件为最佳模型权重，`.csv` 文件记录每轮 epoch 的训练损失、验证损失、验证准确率、验证 AUC 和单轮耗时。

## 命令行推荐

指定测试样本进行 Top-5 推荐演示：

```bash
python main.py ml_recommend --model_path ml1m_dual_target_try5.pt --task_type implicit --neg_ratio 1 --max_rows 500000 --embedding_size 24 --hidden_dims 256,128 --test_id 25 --top_n 5
```

通用推荐指令如下，其中 `x` 表示测试样本 ID，`y` 表示需要返回的推荐电影数量：

```bash
python main.py ml_recommend --model_path ml1m_dual_target_try5.pt --task_type implicit --neg_ratio 1 --max_rows 500000 --embedding_size 24 --hidden_dims 256,128 --test_id x --top_n y
```

例如：

```bash
python main.py ml_recommend --model_path ml1m_dual_target_try5.pt --task_type implicit --neg_ratio 1 --max_rows 500000 --embedding_size 24 --hidden_dims 256,128 --test_id 120 --top_n 5
```

## 交互式网页演示

启动本地推荐网页：

```bash
python recommend_ui.py
```

启动后在浏览器中打开：

```text
http://127.0.0.1:7860
```

网页支持：

- 输入测试样本 ID
- 选择 Top-N 推荐数量
- 随机抽取测试样本
- 展示当前样本电影、用户 ID、推荐电影名、电影类型和模型预测分数
- 显示模型 AUC、电影数量和测试样本数量

## 实验结果

当前模型在验证集上的最佳结果为：

```text
best_val_auc = 0.813159
```

训练过程记录保存在 `ml1m_dual_target_try5.csv`，可用于绘制 epoch 训练曲线，包括 `train_loss`、`val_loss`、`val_acc` 和 `val_auc` 的变化情况。

## 参考资料

- DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
- MovieLens 1M Dataset
- PyTorch 官方文档
