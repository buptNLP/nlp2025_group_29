


#new_predict  零样本，多窗口
import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName
from lag_llama.gluon.estimator import LagLlamaEstimator
import warnings
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.distributions.distribution_output import DistributionOutput

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. 配置参数
TARGET_COL = "OT"
PRED_LENGTH = 24  #预测长度
CONTEXT_LENGTH = PRED_LENGTH * 3  #历史长度
NUM_SAMPLES = 20
NUM_WINDOWS = 3  #画图个数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 数据预处理函数
def preprocess_data(file_path, target_col=TARGET_COL):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    mean = df[target_col].mean()
    std = df[target_col].std()
    df[target_col] = ((df[target_col] - mean) / std).astype(np.float32)
    return df[[target_col]], mean, std

# 3. 加载数据
full_data, mean, std = preprocess_data("ETTh1.csv")

# 4. 安全加载模型
def safe_load_model(device):
    torch.serialization.add_safe_globals([StudentTOutput, DistributionOutput])
    return torch.load("lag-llama.ckpt", map_location=device, weights_only=False)

# 5. 预测函数
def run_lag_llama_predictions(dataset, prediction_length, context_length, num_samples, device):
    ckpt = safe_load_model(device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        nonnegative_pred_samples=True,
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        },
        batch_size=32,
        num_parallel_samples=num_samples,
    )

    predictor = estimator.create_predictor(
        estimator.create_transformation(),
        estimator.create_lightning_module()
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    return list(forecast_it), list(ts_it)

# 6. 滚动预测
def rolling_predictions(data, num_windows=NUM_WINDOWS):
    results = []
    total_length = len(data)

    for i in range(num_windows):
        start_idx = i * PRED_LENGTH
        end_idx = start_idx + CONTEXT_LENGTH + PRED_LENGTH

        if end_idx > total_length:
            print(f"Window {i+1} skipped - Not enough data")
            break

        print(f"\nProcessing window {i+1}/{num_windows}")

        # 准备数据
        window_data = data.iloc[start_idx:end_idx]
        window_ds = ListDataset(
            [{
                FieldName.START: window_data.index[0],
                FieldName.TARGET: window_data[TARGET_COL].values,
            }],
            freq="h"
        )

        # 预测
        forecast, actual = run_lag_llama_predictions(
            window_ds,
            prediction_length=PRED_LENGTH,
            context_length=CONTEXT_LENGTH,
            num_samples=NUM_SAMPLES,
            device=DEVICE
        )
        results.append((forecast[0], actual[0]))

        # 可视化
        plot_window_results(
            data.iloc[:start_idx+CONTEXT_LENGTH],
            data.iloc[start_idx+CONTEXT_LENGTH:end_idx],
            forecast[0],
            window_num=i+1,
            mean=mean,
            std=std
        )
    return results

# 7. 可视化
def plot_window_results(history, actual, forecast, window_num, mean, std):
    plt.figure(figsize=(15, 6))

    # 反标准化
    def inverse_transform(data):
        return data * std + mean

    # 准备数据
    history_values = inverse_transform(history[TARGET_COL].values[-CONTEXT_LENGTH:])
    actual_values = inverse_transform(actual[TARGET_COL].values)
    forecast_mean = inverse_transform(forecast.mean)

    # 时间轴
    history_dates = history.index[-CONTEXT_LENGTH:]
    forecast_dates = pd.date_range(
        start=history.index[-1] + pd.Timedelta(hours=1),
        periods=PRED_LENGTH,
        freq="h"
    )

    # 绘图
    plt.plot(history_dates, history_values, label="History", color="blue")
    plt.plot(forecast_dates, actual_values, label="Actual", color="orange")
    plt.plot(forecast_dates, forecast_mean, label="Forecast", color="green")
    plt.fill_between(
        forecast_dates,
        inverse_transform(forecast.quantile(0.1)),
        inverse_transform(forecast.quantile(0.9)),
        color='green',
        alpha=0.2,
        label="80% CI"
    )

    plt.title(f"Window {window_num} Forecast")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.tight_layout()
    plt.show()

# 8. 执行预测
print("Starting rolling predictions...")
all_results = rolling_predictions(full_data)

# 9. 评估（修正后的关键部分）
if len(all_results) > 0:
    evaluator = Evaluator()
    metrics_list = []

    for forecast, actual in all_results:
        forecast_it = iter([forecast])
        ts_it = iter([actual])
        metrics, _ = evaluator(ts_it, forecast_it)
        metrics_list.append(metrics)

    # 合并指标
    combined_metrics = {}
    for metric_name in metrics_list[0].keys():
        if not metric_name.startswith('MSE'):
            combined_metrics[metric_name] = np.mean([m[metric_name] for m in metrics_list])

    print("\nCombined Evaluation Metrics:")
    for k, v in combined_metrics.items():
        print(f"{k}: {v:.4f}")
else:
    print("No valid predictions for evaluation.")
















#######微调





#new_predict  with fine_tune
import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from lag_llama.gluon.estimator import LagLlamaEstimator
import warnings
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.distributions.distribution_output import DistributionOutput

# 忽略警告
warnings.filterwarnings("ignore")

# 1. 配置参数
TARGET_COL = "OT"
PRED_LENGTH = 24
CONTEXT_LENGTH = PRED_LENGTH * 3
TEST_SIZE = PRED_LENGTH * 2

# 2. 数据预处理函数
def preprocess_data(file_path, target_col=TARGET_COL):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # 数据标准化并转换为float32
    mean = df[target_col].mean()
    std = df[target_col].std()
    df[target_col] = ((df[target_col] - mean) / std).astype(np.float32)

    return df[[target_col]], mean, std

# 3. 加载并预处理数据
data, mean, std = preprocess_data("ETTh1.csv")

# 4. 分割数据集
train_data = data.iloc[:-TEST_SIZE]
test_data = data.iloc[-TEST_SIZE:]

# 5. 创建GluonTS数据集
train_ds = PandasDataset.from_long_dataframe(
    train_data.reset_index().assign(item_id="series1"),
    target=TARGET_COL,
    timestamp="date",
    freq="h",
    item_id="item_id"
)

test_ds = ListDataset(
    [{
        "start": test_data.index[0],
        "target": test_data[TARGET_COL].values.astype(np.float32)
    }],
    freq="h"
)

# 6. 修正模型配置
def get_model_config(device):
    torch.serialization.add_safe_globals([StudentTOutput, DistributionOutput])
    ckpt = torch.load("lag-llama.ckpt", map_location=device, weights_only=False)

    model_args = ckpt["hyper_parameters"]["model_kwargs"]

    model_args.update({
        "input_size": 1,
        "time_feat": True,
        "n_layer": 12,
        "n_head": 12,
        "n_embd_per_head": 144 // 12,
        "scaling": "std",
        "context_length": CONTEXT_LENGTH
    })

    return model_args

# 7. 创建估计器
def create_estimator(model_args):
    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=PRED_LENGTH,
        context_length=model_args["context_length"],
        input_size=model_args["input_size"],
        n_layer=model_args["n_layer"],
        n_embd_per_head=model_args["n_embd_per_head"],
        n_head=model_args["n_head"],
        scaling=model_args["scaling"],
        time_feat=model_args["time_feat"],
        nonnegative_pred_samples=True,
        batch_size=8,
        num_parallel_samples=20,
        lr=1e-4,
        trainer_kwargs={"max_epochs": 5, "accelerator": "auto"}  # 修改为1个epoch
    )

    # 强制设置模型为float32
    estimator.create_lightning_module().model = estimator.create_lightning_module().model.float()
    return estimator

# 8. 主流程
device = "cuda" if torch.cuda.is_available() else "cpu"
model_args = get_model_config(device)
estimator = create_estimator(model_args)

# 9. 微调模型
print("Fine-tuning model (10 epochs)...")
predictor = estimator.train(train_ds)

# 10. 预测和可视化
def inverse_transform(data, mean, std):
    return data * std + mean

def plot_results(train, test, forecast, mean, std):
    plt.figure(figsize=(15, 6))

    # 反标准化
    train = inverse_transform(train, mean, std)
    test = inverse_transform(test, mean, std)
    forecast_mean = inverse_transform(forecast.mean, mean, std)
    forecast_low = inverse_transform(forecast.quantile(0.1), mean, std)
    forecast_high = inverse_transform(forecast.quantile(0.9), mean, std)

    # 绘图
    plt.plot(train.index[-CONTEXT_LENGTH:], train[-CONTEXT_LENGTH:], label="History")
    test_dates = pd.date_range(
        start=train.index[-1] + pd.Timedelta(hours=1),
        periods=len(test),
        freq="h"
    )
    plt.plot(test_dates, test, label="Actual")

    forecast_dates = test_dates[:PRED_LENGTH]
    plt.plot(forecast_dates, forecast_mean, label="Forecast")
    plt.fill_between(forecast_dates, forecast_low, forecast_high, alpha=0.2)

    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.title(f"ETTh1 {TARGET_COL} Forecast (10 epochs)")
    plt.tight_layout()
    plt.show()

# 11. 运行预测
print("Running predictions...")
forecasts, tss = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)
forecasts = list(forecasts)
tss = list(tss)

# 12. 可视化结果
plot_results(train_data[TARGET_COL], test_data[TARGET_COL].values, forecasts[0], mean, std)

# 13. 评估
evaluator = Evaluator()
metrics, _ = evaluator(tss, forecasts)
print("Evaluation Metrics (10 epochs):", {k: round(v, 4) for k, v in metrics.items()})
