from peft import LoraConfig, get_peft_model, PeftModel
import re
from transformers import AutoModel
import numpy as np
from transformers import BertTokenizerFast, BertModel
import argparse
import torch
from pre.data_factory import data_provider
from pre.Autoformer_EncDec import series_decomp
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

class TimeSeriesProcessor:
    def __init__(self, tokenizer, dataset, seq_len, pred_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dataset = dataset
        self.descr = {
                 'ETTh1.csv': "Describes hourly data. Each of them contains load characteristics of seven oil and power transformers.",
                 'ETTh2.csv': "Describes hourly data. Each of them contains load characteristics of seven oil and power transformers.",
                 'ETTm1.csv': "Describes minute-level data. Each of them contains seven load characteristics of oil and power transformers.",
                 'ETTm2.csv': "Describes minute-level data. Each of them contains seven load characteristics of oil and power transformers.",
                 'traffic.csv': "Describes road occupancy. It contains hourly data recorded by sensors on San Francisco highways from 2015 to 2016.",
                 'electricity.csv': "Describes the hourly electricity consumption of 321 customers collected from 2012 to 2014.",
                 'exchange_rate.csv': "Describes the daily exchange rates of eight countries from 1990 to 2016.",
                 'weather.csv': "It includes 21 weather indicators, such as air temperature and humidity. Its data is recorded every 10 minutes in 2020.",
                 'national_illness': "Describes the ratio of patients with influenza to the total number of patients. It includes weekly data from the US Centers for Disease Control and Prevention from 2002 to 2021."
                 }

    def ts_to_text(self, batch_x, batch_x_mark):
        """ 时间序列转自然语言描述 """
        batch_x_f = ''
        for x in batch_x:
            batch_x_f += str(f"{x.item():.2f}")
            batch_x_f += ' '

        batch_x_mark_f = ''
        for x in batch_x_mark:
            batch_x_mark_f += str(f"{x[0].item():.2f}")
            batch_x_mark_f += ' '

        des = f'You are a time series forecasting expert tasked with predicting future trends based on historical data. You need to predict the future trend of {self.pred_len} units based on the given {self.seq_len} units of historical time series data (e.g., daily sales, temperature, or stock prices) '# and features (including time-related information such as hours, days of the week, months, etc., including periodicity). First, preprocess the data, including handling missing values and outliers, and extract key features (such as moving window statistics, periodic features, etc.); then select an appropriate forecasting model, automatically optimize parameters, and output future forecast results. If the data is insufficient or of poor quality, inferences must be made based on other information. Below is the data description:\n'
        des += self.descr[self.dataset]
        des += '\n'
        des += f'Complete the prediction task, and the final result is a prediction sequence of {self.pred_len} units in length.'
        des += '\n'
        des += 'Time series data:'
        des += batch_x_f
        des += '\n'
        # des += 'Corresponded several time feature data:'
        # des += '\n'
        # des += batch_x_mark_f

        return des

        # time_feats = []
        # for t in range(len(batch_x)):
        #     # 提取时间特征（假设batch_x_mark包含[hour, weekday]）
        #     hour = int(batch_x_mark[t, 0])
        #     weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(batch_x_mark[t, 1])]
        #     time_desc = f"{weekday} {hour}:00"
        #     value = batch_x[t, 0]
        #     time_feats.append(f"Time {time_desc}: {value:.2f}")
        #
        # prompt = "Forecast next values based on:\n" + "\n".join(time_feats)
        # return prompt

    def format_target(self, batch_y):
        """ 格式化预测目标 """
        t = ''
        for x in batch_y:
            t += str(f"{x.item():.1f}")
            t += ' '
        return t

    def prepare_llm_input(self, batch):
        """ 处理完整batch数据 """
        prompt = self.ts_to_text(batch[0], batch[2])
        print(len(prompt))
        target = self.format_target(batch[1])

        # Tokenization
        model_inputs = self.tokenizer(
            prompt,
            max_length=train_args['token_size'],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # 标签token化
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target,
                max_length=train_args['token_size'],
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
        print(labels.shape)
        return {
            "input_ids": model_inputs.input_ids[0, :],
            "attention_mask": model_inputs.attention_mask[0, :],
            "labels": labels[0, :]
        }

# class TimeSeriesProcessor:
#     def __init__(self, tokenizer, dataset, seq_len=96, pred_len=96):
#         self.tokenizer = tokenizer
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.dataset = dataset
#         self.descr = {
#             'ETTh1.csv': "描述了一个小时级数据。它们中的每一个都包含七种石油和电力变压器的负载特征。",
#             'ETTh2.csv': "描述了一个小时级数据。它们中的每一个都包含七种石油和电力变压器的负载特征。",
#             'ETTm1.csv': "描述了一个分钟级数据。它们中的每一个都包含七种石油和电力变压器的负载特征。",
#             'ETTm2.csv': "描述了一个小时级数据。它们中的每一个都包含七种石油和电力变压器的负载特征。",
#             'traffic.csv': "描述了道路占用率。它包含 2015 年至 2016 年旧金山高速公路传感器记录的每小时数据。",
#             'electricity.csv': "描述了从 2012 年到 2014 年收集了 321 个客户每小时电力消耗。",
#             'exchange_rate.csv': "描述了 1990 年至 2016 年 8 个国家的每日汇率。",
#             'weather.csv': "包括 21 个天气指标，例如空气温度和湿度。它的数据在 2020 年的每 10 分钟记录一次。",
#             'national_illness': "描述了患有流感疾病的患者与患者数量的比率。它包括 2002 年至 2021 年美国疾病控制和预防中心每周数据。"
#         }
#
#     def ts_to_text(self, batch_x, batch_x_mark):
#         """时间序列转自然语言描述，确保输入为 [batch_size, seq_len]"""
#         if batch_x.dim() == 3:  # 如果是 [batch_size, seq_len, features]
#             batch_x = batch_x[:, :, 0]  # 取第一个特征（假设单变量）
#
#         des = (
#             f'任务描述：你是一个时间序列预测专家，需基于历史数据预测未来趋势。'
#             f'你需要根据给定的 {self.seq_len} 单位长度的历史时间序列数据（如每日销售额、气温或股票价格）'
#             f'和特征（包含时间信息如小时、星期、月份等，包括周期性），预测未来 {self.pred_len} 单位的变化趋势。'
#             '首先对数据进行预处理，包括处理缺失值和异常值，并提取关键特征（如滑动窗口统计量、周期特征等）；'
#             '然后选择合适的预测模型，自动优化参数并输出未来的预测结果。'
#             '若数据不足或质量较差，需根据其他信息进行推断。\n'
#             f'数据描述：{self.descr.get(self.dataset, "未知数据集")}\n'
#             f'时间序列数据（前5个值）：{batch_x[:, :5].tolist()}\n'
#             f'时间特征（前5个）：{batch_x_mark[:, :5].tolist()}\n'
#             f'预测目标长度：{self.pred_len}'
#         )
#         return des
#
#     def format_target(self, batch_y):
#         """格式化预测目标为字符串，输入应为 [batch_size, pred_len]"""
#         if batch_y.dim() == 3:  # 如果是 [batch_size, pred_len, features]
#             batch_y = batch_y[:, :, 0]  # 取第一个特征
#         return " ".join([f"{x:.2f}" for x in batch_y[0]])  # 取batch中第一个样本
#
#     def prepare_llm_input(self, batch):
#         """处理完整batch数据，确保输出维度为 [batch_size, ...]"""
#         batch_x, batch_y, batch_x_mark, batch_y_mark = batch  # 假设batch是元组 (x, y, x_mark)
#
#         # 检查并调整输入维度
#         print(batch_x.shape)
#         if batch_x.dim() == 3 and batch_x.size(0) != args.batch_size:
#             batch_x = batch_x.permute(1, 0, 2)  # [seq_len, batch_size, features] -> [batch_size, seq_len, features]
#
#         prompts = []
#         labels = []
#         for i in range(batch_x.size(0)):  # 遍历batch中的每个样本
#             prompt = self.ts_to_text(batch_x[i].unsqueeze(0), batch_x_mark[i].unsqueeze(0))  # 保持二维输入
#             target = self.format_target(batch_y[i].unsqueeze(0))
#             prompts.append(prompt)
#             labels.append(target)
#
#         # Tokenization (确保返回 [batch_size, seq_len])
#         model_inputs = self.tokenizer(
#             prompts,
#             max_length=1024,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#
#         with self.tokenizer.as_target_tokenizer():
#             labels = self.tokenizer(
#                 labels,
#                 max_length=128,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt"
#             ).input_ids
#
#         return {
#             "input_ids": model_inputs.input_ids,  # [batch_size, seq_len]
#             "attention_mask": model_inputs.attention_mask,  # [batch_size, seq_len]
#             "labels": labels  # [batch_size, pred_len]
#         }


class TimeTrainSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data_loader, processor):
        self.loader = data_loader
        self.processor = processor

    def __len__(self):
        return 400
        # return len(self.loader.dataset) // 10

    def __getitem__(self, idx):
        batch = self.loader.dataset[idx]
        return self.processor.prepare_llm_input(batch)


class TimeEvalSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data_loader, processor):
        self.loader = data_loader
        self.processor = processor

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        batch = self.loader.dataset[idx]
        return self.processor.prepare_llm_input(batch)


# 打印可训练参数占比
def print_trainable_params(model):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"可训练参数: {trainable} | 总参数: {total} | 占比: {100 * trainable / total:.2f}%")


# 自定义数据整理器
def data_collator(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features])
    }


# 自定义损失计算函数
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.argmax(preds, axis=-1)
    print(preds)

    # 解码预测结果
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # 替换-100为pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 提取数值部分
    def extract_numbers(text):
        return [float(x) for x in re.findall(r"\d+\.\d+", text)]

    true_values = [extract_numbers(l) for l in decoded_labels]
    pred_values = [extract_numbers(p) for p in decoded_preds]
    true_values = [l[:train_args['pred_len'] - 2] for l in true_values]
    pred_values = [p[:train_args['pred_len'] - 2] for p in pred_values]
    print(true_values, pred_values)


    # 计算MAE/MSE
    mae = np.mean([np.abs(np.array(p) - np.array(t)) for p, t in zip(pred_values, true_values)])
    mse = np.mean([(np.array(p) - np.array(t)) ** 2 for p, t in zip(pred_values, true_values)])

    return {"MAE": mae, "MSE": mse}


# 参数


train_args = {'llm_name': 'qwen3-0.6b',
              'batch_size': 1,
              'seq_len': 48,
              'pred_len': 12,
              'token_size': 768}


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


args = parse_args()

# 1. 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 使用4位量化
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. 加载模型和tokenizer
tokenizer = BertTokenizerFast.from_pretrained(train_args['llm_name'])
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

model = AutoModelForCausalLM.from_pretrained(
    train_args['llm_name'],
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 添加新token
model.resize_token_embeddings(len(tokenizer))  # 调整模型embedding层
# model_name = "t5-base"  # 可选：t5-base, t5-large, google/flan-t5-base
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# 初始化时间序列处理器
ts_processor = TimeSeriesProcessor(tokenizer, args.data_path, seq_len=train_args['seq_len'], pred_len=train_args['pred_len'])

# 获取数据加载器
train_dataset, train_loader = data_provider(args, 'train')
eval_data, eval_loader = data_provider(args, 'val')
# for batch_idx, (a, b, c, d) in enumerate(train_loader):
#     print(a.shape, b.shape, c.shape, d.shape)

# 转换第一个batch为LLM输入格式
# sample_batch = next(iter(train_loader))
# llm_input = ts_processor.prepare_llm_input(sample_batch)

processed_train_dataset = TimeTrainSeriesDataset(train_loader, ts_processor)
processed_eval_dataset = TimeEvalSeriesDataset(eval_loader, ts_processor)


# print("input_ids shape:", processed_train_dataset["input_ids"].shape)
# print("attention_mask shape:", processed_train_dataset["attention_mask"].shape)

# 6. 配置LoRA参数
lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,  # Dropout概率
    bias="none",  # 偏置处理方式
    task_type="CAUSAL_LM"  # 任务类型
)

# 7. 应用LoRA到模型
model = get_peft_model(model, lora_config)

print_trainable_params(model)

# 8. 训练配置
training_args = TrainingArguments(
    output_dir="./ts_forecast",
    gradient_accumulation_steps=4,
    per_device_train_batch_size=train_args['batch_size'],
    per_gpu_eval_batch_size=train_args['batch_size'],
    learning_rate=1e-4,  # 更小的学习率
    num_train_epochs=5,
    logging_steps=50,
    eval_strategy="steps",  # 按步评估
    eval_steps=400,  # 每200步评估一次
    # save_strategy="steps",
    # save_steps=20,
    fp16=True,
    # load_best_model_at_end=True,  # 训练结束时加载最佳模型
    report_to="tensorboard"
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics  # 见下方新增函数
)
trainer.train()

model.save_pretrained("lora_adapter")


