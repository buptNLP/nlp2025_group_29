import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os


# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 数据加载和预处理
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    target = data[['OT']].values
    return data, target

data, target = load_data('./data/ETT/ETTh1.csv')

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler.fit_transform(target)

# 创建时间序列数据集
def create_dataset(data, look_back=24):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

look_back = 24  # 使用24小时(1天)的数据来预测下一小时
X, y = create_dataset(target_scaled, look_back)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为PyTorch张量并调整形状 [samples, seq_len, features]
X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(device)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
X_test = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(device)

# 创建DataLoader
batch_size = 32
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

# 2. 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel().to(device)
print(model)

# 3. 训练模型
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)
    
    # 验证损失
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        val_losses.append(val_loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 绘制训练和验证损失
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 4. 评估和预测
def evaluate_model(model, X, y_true, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    
    # 反标准化
    predictions = scaler.inverse_transform(predictions.cpu().numpy())
    y_true = scaler.inverse_transform(y_true.cpu().numpy())
    
    # 计算评估指标
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mse)
    
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    
    return predictions, y_true, mse, mae, rmse

print("Train Metrics:")
train_pred, train_true, train_mse, train_mae, train_rmse = evaluate_model(model, X_train, y_train, scaler)

print("\nTest Metrics:")
test_pred, test_true, test_mse, test_mae, test_rmse = evaluate_model(model, X_test, y_test, scaler)

# 5. 可视化预测结果
def plot_predictions(true, pred, title):
    plt.figure(figsize=(15, 6))
    plt.plot(true, label='Actual')
    plt.plot(pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('OT Temperature')
    plt.legend()
    plt.show()

# 绘制训练集预测
plot_predictions(train_true, train_pred, 'Training Set Predictions')

# 绘制测试集预测
plot_predictions(test_true, test_pred, 'Test Set Predictions')

# 绘制测试集部分放大图
plt.figure(figsize=(15, 6))
plt.plot(test_true[:200], label='Actual')
plt.plot(test_pred[:200], label='Predicted')
plt.title('Test Set Predictions (First 200 Samples)')
plt.xlabel('Time Steps')
plt.ylabel('OT Temperature')
plt.legend()
plt.show()