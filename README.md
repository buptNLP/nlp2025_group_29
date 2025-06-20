# SeriesLLM 时间序列预测工具包

## 项目概述

SeriesLLM 是一个基于大语言模型(LLM)和传统时间序列模型的时间序列预测工具包，集成了多种先进的预测方法，包括 Lag-Llama、TTM (Time Series Transformer Model)、LSTM 以及通用大语言模型(如 GPT、Llama、DeepSeek 等)的微调方案。

## 目录结构

```
SeriesLLM/
├── data/                   # 数据集文件夹
│   ├── ETTh1.csv          # ETT数据集
│   ├── traffic.csv        # 交通数据集
│   └── ...                # 其他数据集
├── pre/                   # 数据预处理模块
├── ts_forecast/           # 训练和微调数据记录
├── general_llm.py         # 通用大模型训练脚本 (Llama/GPT/DeepSeek等)
├── lag_llama.py           # Lag-Llama 微调脚本
├── lstm_test.py           # LSTM 训练脚本
├── TTM.py                 # TTM 微调脚本
└── README.md              # 使用说明文档
```

## 数据集准备

将 CSV 格式时间序列数据放入 `data/` 目录下。数据应包含时间戳列和目标值列，例如：

```csv
date,OT,HUFL,HULL,MUFL,MULL,LUFL,LULL
2016-07-01 00:00:00,4.1824,4.3956,2.4605,3.1067,1.5580,1.7320,0.9281
2016-07-01 01:00:00,4.1498,4.3915,2.4605,3.0979,1.5580,1.7315,0.9281
```

## 模型使用指南

### 通用大语言模型 (Llama/GPT/DeepSeek)

**文件**: `general_llm.py`

**功能**: 微调通用LLM用于时间序列预测

**使用方法**:

1. 配置训练参数:
   ```python
   train_args = {
       'llm_name': 'qwen3-0.6b',  # 模型名称
       'batch_size': 1,           # 批大小
       'seq_len': 48,             # 序列长度
       'pred_len': 12,            # 预测长度
       'token_size': 768          # token大小
   }
   ```

2. 配置数据预处理参数:
   ```python
    def parse_args():
        parser = argparse.ArgumentParser(description='Time Series Decomposition')
        parser.add_argument('--data', type=str, default='custom', help='dataset name')
        parser.add_argument('--data_path', type=str, default='exchange_rate.csv', help='data file name')
        parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
        parser.add_argument('--features', type=str, default='S', choices=['S', 'M', 'MS'],
                            help='features type: S=univariate, M=multivariate, MS=multivariate with target')
        parser.add_argument('--target', type=str, default='OT', help='target feature')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding: h=hourly, t=minutely')
        parser.add_argument('--seq_len', type=int, default=train_args['seq_len'], help='input sequence length')
        parser.add_argument('--label_len', type=int, default=0, help='start token length')
        parser.add_argument('--pred_len', type=int, default=train_args['pred_len'], help='prediction sequence length')
        parser.add_argument('--batch_size', type=int, default=train_args['batch_size'], help='batch size')
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
        parser.add_argument('--kernel_size', type=int, default=25, help='decomposition kernel size')
        return parser.parse_args()
   ```
   
3. 开始训练

### Lag-Llama 模型

**文件**: `lag_llama.py`

**功能**: 零样本预测和微调 Lag-Llama 模型

**使用方法**:

1. **零样本预测**:
   ```python
   # 配置参数
   TARGET_COL = "OT"  # 目标列名
   PRED_LENGTH = 24   # 预测长度
   CONTEXT_LENGTH = PRED_LENGTH * 3  # 历史长度
   
   # 加载数据
   full_data, mean, std = preprocess_data("ETTh1.csv")
   
   # 执行滚动预测
   all_results = rolling_predictions(full_data)
   ```

2. **微调模式**:
   ```python
   # 分割数据集
   train_data = data.iloc[:-TEST_SIZE]
   test_data = data.iloc[-TEST_SIZE:]
   
   # 创建估计器并微调
   estimator = create_estimator(model_args)
   predictor = estimator.train(train_ds)
   
   # 预测和评估
   forecasts, tss = make_evaluation_predictions(dataset=test_ds, predictor=predictor)
   ```



### TTM 模型

**文件**: `TTM.py`

**功能**: 零样本评估和少量样本微调TTM模型

**使用方法**:

1. **零样本评估**:
   ```python
   zeroshot_eval(
       dataset_name="etth1",
       batch_size=64,
       context_length=512,
       forecast_length=96
   )
   ```

2. **少量样本微调**:
   ```python
   fewshot_finetune_eval(
       dataset_name="etth1",
       context_length=512,
       forecast_length=96,
       batch_size=64,
       fewshot_percent=5,  # 使用5%的数据微调
       learning_rate=0.001
   )
   ```

### 4. LSTM 模型

**文件**: `lstm_test.py`

**功能**: 传统LSTM模型训练和预测

**使用方法**:

1. 加载和预处理数据:
   ```python
   data, target = load_data('./data/ETT/ETTh1.csv')
   scaler = MinMaxScaler(feature_range=(0, 1))
   target_scaled = scaler.fit_transform(target)
   ```

2. 训练模型:
   ```python
   model = LSTMModel().to(device)
   criterion = nn.MSELoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   for epoch in range(num_epochs):
       # 训练代码...
   ```

3. 评估和预测:
   ```python
   evaluate_model(model, X_test, y_test, scaler)
   ```

