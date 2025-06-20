import argparse
import torch
from data_factory import data_provider
from Autoformer_EncDec import series_decomp


def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Decomposition')
    parser.add_argument('--data', type=str, required=True, help='dataset name')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True, help='data file name')
    parser.add_argument('--features', type=str, default='S', choices=['S', 'M', 'MS'],
                        help='features type: S=univariate, M=multivariate, MS=multivariate with target')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding: h=hourly, t=minutely')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition kernel size')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载数据
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # 2. 初始化序列分解模块
    decomp = series_decomp(args.kernel_size)

    # 3. 处理训练数据
    print("\nProcessing training data...")
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        # batch_x shape: [batch_size, seq_len, feature_dim]
        seasonal, trend = decomp(batch_x)

        print(f"Batch {i + 1}:")
        print(f"Original x data shape: {batch_x[0, :, 0]}")
        print(f"Original x data shape: {batch_x_mark[0, :, 0]}")
        #print(f"Original y data shape: {batch_y[0, :, 0]}")
        #print(f"Original x data shape: {batch_y_mark[0, :, 0]}")
        # print(f"Seasonal component shape: {seasonal.shape}")
        #         # print(f"Trend component shape: {trend.shape}")

        # 只展示前3个batch
        if i == 0:
            break

    # 4. 可选: 保存分解后的数据
    save_decomposed_data = True
    if save_decomposed_data:
        torch.save({
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'decomp': decomp.state_dict()
        }, 'decomposed_data.pth')


if __name__ == '__main__':
    main()

# 参数说明：
#
# --data: 数据集名称 (ETTh1, ETTh2, ETTm1, ETTm2 或 custom)
#
# --root_path: 数据文件根目录
#
# --data_path: 数据文件名
#
# --kernel_size: 分解核大小 (默认25)
#
# python preprocess.py --data ETT --root_path ../data --data_path ETTh1.csv --kernel_size 25