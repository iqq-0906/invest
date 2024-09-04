# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
# import jax
import numpy as np
# from jax import random, jit
from kan import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import sklearn
from sklearn.preprocessing import MinMaxScaler
import itertools
from functools import partial
from tqdm import trange, tqdm
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time
# from torch.func import jacrev, hessian
from functorch import jacrev, hessian
import pandas as pd
# from Covariance_estimation.gerber import gerber_cov_stat1, gerber_cov_stat2,is_psd_def
# from Covariance_estimation.ledoit import ledoit


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class r_th(nn.Module):
    def __init__(self, model):
        super(r_th, self).__init__()
        set_seed(10)
        self.model = model
        self.device = next(model.parameters()).device  # Get the device of the model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Initialize optimizer

    def reshape(self, X):
        return X.reshape(-1, )

    def helper(self, *xs):
        reshaped_xs = [self.reshape(x) for x in xs]
        stacked_tensor = torch.stack(reshaped_xs)
        return stacked_tensor.permute(1, 0)

    def operator_net(self, *xs):
        y = self.helper(*xs)
        return self.model(y)

    def residual_net(self, *xs):
        s = self.operator_net(*xs)
        s_gradients = [jacrev(self.operator_net, argnums=i)(*xs).sum(dim=0) for i in range(len(xs))]
        
        # Apply ReLU to gradients
        s_gradients = [torch.relu(grad) for grad in s_gradients]

        return s_gradients

    def residual_net1(self, *xs, ones):
        s_gradients = self.residual_net(*xs)
        output = sum(s_gradients)
        return torch.mean((output.flatten() - ones) ** 2)

    def residual_net2(self, *xs):
        r = self.operator_net(*xs)
        return -torch.mean(r.flatten())

    def residual_net3(self, *xs):
        s = self.operator_net(*xs)
        s_gradients = self.residual_net(*xs)
        s1 = sum(g * x for g, x in zip(s_gradients, xs))
        r_mean = torch.mean(s1.flatten())
        th = self.residual_net42(*xs)
        return 1.25 * th - r_mean

    def residual_net42(self, *xs):
        r = self.operator_net(*xs)
        return torch.std(r.flatten())

    def residual_net4(self, *xs):
        r = self.operator_net(*xs)
        r_mean = torch.mean(r.flatten())
        th = self.residual_net42(*xs)
        return -r_mean / th

    def residual_net5(self, *xs):
        s = self.operator_net(*xs)
        s_gradients = self.residual_net(*xs)
        s1 = sum(g * x for g, x in zip(s_gradients, xs))
        return torch.mean((s1.flatten() - s.flatten()) ** 2)

    def calculate_cvar(self, r, alpha=0.05):
        var = torch.quantile(r, 1 - alpha)
        cvar = torch.mean(r[r >= var])
        return torch.mean((cvar.flatten()) ** 2)

    def residual_net_cvar(self, *xs, alpha):
        r = self.operator_net(*xs)
        return self.calculate_cvar(r, alpha)

    def train(self, *xs, ones, covariance_matrix=None):
        # self.optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, history_size=10, line_search_fn="strong_wolfe",
        #                   tolerance_grad=1e-32, tolerance_change=1e-32)
        self.optimizer= torch.optim.Adam(model.parameters(), lr=0.001)


        pbar = tqdm(range(30), desc='description')
        for _ in pbar:
            self.optimizer.zero_grad()

            # 计算各个损失
            loss1 = self.residual_net1(*xs,ones= ones)
            loss2 = self.residual_net2(*xs)
            loss3 = self.residual_net3(*xs)
            loss4 = self.residual_net4(*xs)
            loss5 = self.residual_net5(*xs)
            # if loss2.item() > 0 :
            #     continue
            #     # 跳过当前 epoch，重新训练

            # 合并损失函数，权重根据实际情况调整
            loss =loss1  +loss3+loss5

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            if _ % 1 == 0:
                pbar.set_description("loss1: %.2e | loss3: %.2e| loss5: %.2e" %
                                     (loss1.cpu().detach().numpy(), loss3.cpu().detach().numpy(),
                                      loss5.cpu().detach().numpy()
                                      ))




i=64
metrics_df = pd.DataFrame()
for l in range(1, 6):
    pre_data=pd.read_csv('tly/BL_returnBL_rolling{}_{}.csv'.format(l,i))
    print(pre_data)
    df = pd.DataFrame()
    for j in range(0,64,i):
        print('{}-{}-{}'.format(i,l,j))
        if i in [5,10,15,20,30,64]:
            if j != 60:
                pre_data_return=pre_data.iloc[j:j+i,:]
                cov_data=pd.read_csv('iqq/BL_cov_rolling{}_{}_{}.csv'.format(l,i,j))

                del cov_data['Unnamed: 0']
                covariance_matrix = torch.tensor(cov_data.values).float().to(device)
                ones = torch.ones(i).to(device)
                tensor_list = []
                for m in range(10):
                    numpy_array = pre_data_return.iloc[:, m].to_numpy(dtype=np.float64)
                    tensor_name = torch.tensor(numpy_array, dtype=torch.float64).reshape(-1, ).float().to(device)
                    torch.set_printoptions(precision=6)
                    tensor_list.append(tensor_name)
                kan  = KAN(width=[4,2,1,1], grid=3, k=3, seed=1, device=device)
                model = r_th(kan)
                model.train(tensor_list[0],tensor_list[1],tensor_list[2],tensor_list[3],tensor_list[4],tensor_list[5],
                tensor_list[6],tensor_list[7],tensor_list[8],tensor_list[9],ones=ones,covariance_matrix=covariance_matrix)
                proportion=model.residual_net(tensor_list[0],tensor_list[1],tensor_list[2],tensor_list[3],tensor_list[4],tensor_list[5],
                tensor_list[6],tensor_list[7],tensor_list[8],tensor_list[9])
                n=0
                list=[]
                df1=pd.DataFrame()
                for t in proportion:
                    n+=1
                    print("s_x{}".format(n),t)
                    numpy_data = t.view(-1).tolist()
                    numpy_data=np.array(numpy_data)
                    # print(numpy_data)
                    df1['{}'.format(n)]=numpy_data
                df= pd.concat([df, df1], axis=0, ignore_index=True)
            if j==60:
                pre_data_return = pre_data.iloc[j:j + i, :]
                cov_data = pd.read_csv('data/BL_cov_rolling{}_{}_{}.csv'.format(l, i,j))
                del cov_data['Unnamed: 0']
                covariance_matrix = torch.tensor(cov_data.values).float()
                ones = torch.ones(4)
                tensor_list = []
                for m in range(10):
                    numpy_array = pre_data_return.iloc[:, m].to_numpy(dtype=np.float64)
                    tensor_name = torch.tensor(numpy_array, dtype=torch.float64).reshape(-1, ).float()
                    torch.set_printoptions(precision=6)
                    tensor_list.append(tensor_name)
                kan = KAN(width=[4,2,1,1], grid=3, k=3, seed=1, device=device)
                model = r_th(kan)
                model.train(tensor_list[0], tensor_list[1], tensor_list[2], tensor_list[3], tensor_list[4],
                            tensor_list[5],
                            tensor_list[6], tensor_list[7], tensor_list[8], tensor_list[9], ones=ones,covariance_matrix=covariance_matrix)
                proportion = model.residual_net(tensor_list[0], tensor_list[1], tensor_list[2], tensor_list[3],
                                                tensor_list[4], tensor_list[5],
                                                tensor_list[6], tensor_list[7], tensor_list[8], tensor_list[9])
                n = 0
                list = []
                df1 = pd.DataFrame()
                for t in proportion:
                    n += 1
                    print("s_x{}".format(n), t)
                    numpy_data = t.view(-1).tolist()
                    numpy_data = np.array(numpy_data)
                    print(numpy_data)
                    df1['{}'.format(n)] = numpy_data
                df = pd.concat([df, df1], axis=0, ignore_index=True)

        if i in [40]:
            if j==0:
                pre_data_return = pre_data.iloc[j:j + i, :]
                cov_data = pd.read_csv('data/BL_cov_rolling{}_{}_{}.csv'.format(l, i, j))
                del cov_data['Unnamed: 0']
                covariance_matrix = torch.tensor(cov_data.values).float()
                ones = torch.ones(i)
                tensor_list = []
                for m in range(10):
                    numpy_array = pre_data_return.iloc[:, m].to_numpy(dtype=np.float64)
                    tensor_name = torch.tensor(numpy_array, dtype=torch.float64).reshape(-1, ).float()
                    torch.set_printoptions(precision=6)
                    tensor_list.append(tensor_name)
                kan =KAN(width=[4,2,1,1], grid=3, k=3, seed=1, device=device)
                model = r_th(kan)
                model.train(tensor_list[0], tensor_list[1], tensor_list[2], tensor_list[3], tensor_list[4], tensor_list[5],
                            tensor_list[6], tensor_list[7], tensor_list[8], tensor_list[9], ones=ones,covariance_matrix=covariance_matrix)
                proportion = model.residual_net(tensor_list[0], tensor_list[1], tensor_list[2], tensor_list[3],
                                                tensor_list[4], tensor_list[5],
                                                tensor_list[6], tensor_list[7], tensor_list[8], tensor_list[9])
                n = 0
                list = []
                df1 = pd.DataFrame()
                for t in proportion:
                    n += 1
                    print("s_x{}".format(n), t)
                    numpy_data = t.view(-1).tolist()
                    numpy_data = np.array(numpy_data)
                    print(numpy_data)
                    df1['{}'.format(n)] = numpy_data
                df = pd.concat([df, df1], axis=0, ignore_index=True)
            else:
                pre_data_return = pre_data.iloc[j:j + i, :]
                cov_data = pd.read_csv('data/BL_cov_rolling{}_{}_{}.csv'.format(l, i, j))
                del cov_data['Unnamed: 0']
                covariance_matrix = torch.tensor(cov_data.values).float()
                ones = torch.ones(24)
                tensor_list = []
                for m in range(10):
                    numpy_array = pre_data_return.iloc[:, m].to_numpy(dtype=np.float64)
                    tensor_name = torch.tensor(numpy_array, dtype=torch.float64).reshape(-1, ).float()
                    torch.set_printoptions(precision=6)
                    tensor_list.append(tensor_name)
                kan = KAN(width=[4,2,1,1], grid=3, k=3, seed=1, device=device)
                model = r_th(kan)
                model.train(tensor_list[0], tensor_list[1], tensor_list[2], tensor_list[3], tensor_list[4],
                            tensor_list[5],
                            tensor_list[6], tensor_list[7], tensor_list[8], tensor_list[9], ones=ones,covariance_matrix=covariance_matrix)
                proportion = model.residual_net(tensor_list[0], tensor_list[1], tensor_list[2], tensor_list[3],
                                                tensor_list[4], tensor_list[5],
                                                tensor_list[6], tensor_list[7], tensor_list[8], tensor_list[9])
                n = 0
                list = []
                df1 = pd.DataFrame()
                for t in proportion:
                    n += 1
                    print("s_x{}".format(n), t)
                    numpy_data = t.view(-1).tolist()
                    numpy_data = np.array(numpy_data)
                    print(numpy_data)
                    df1['{}'.format(n)] = numpy_data
                df = pd.concat([df, df1], axis=0, ignore_index=True)
    df.to_csv('quanzhong_rolling_{}_{}.csv'.format(i,l))
for g in range(1, 6):
    period_weights_df=pd.read_csv('quanzhong_rolling_{}_{}.csv'.format(i,g))
    del period_weights_df['Unnamed: 0']
    # 计算每行的总和
    row_sums =period_weights_df.sum(axis=1)
    # 归一化每行的权重
    period_weights_df=period_weights_df.div(row_sums, axis=0)
    def compute_period_weights(weights_df, period):
        num_days = len(weights_df)
        period_weights = np.zeros_like(weights_df.values)

        # 按每个周期计算均值
        for start_day in range(0, num_days, period):
            end_day = min(start_day + period, num_days)
            avg_weights = weights_df.iloc[start_day:end_day].mean().values
            period_weights[start_day:end_day] = avg_weights

        return pd.DataFrame(period_weights, columns=weights_df.columns)
    holding_period =g
    period_weights_df= compute_period_weights(period_weights_df, holding_period)
    period_weights_df =period_weights_df.to_numpy()
    true_data = pd.read_csv('iqq/stock_true_data_rolling{}_{}.csv'.format(g,i)).to_numpy()
    product_df =period_weights_df*true_data
    product_df= pd.DataFrame(product_df)
    row_sum = product_df.sum(axis=1)
    # 将结果转换为 DataFrame 或 Series
    row_sum_df = pd.DataFrame(row_sum, columns=['returns'])
    total_sum = row_sum_df['returns'].sum()
    # print("Total Sum of Returns:", total_sum)
    daily_returns = row_sum_df
    # daily_returns.to_csv('daily_returns_BL64.csv', index=False)
    def calculate_performance_metrics(daily_returns, risk_free_rate=0):
        daily_returns_df=pd.DataFrame()
        if 'returns' not in daily_returns.columns:
            raise ValueError("DataFrame must contain a 'returns' column")
        #计算实际收益率
        daily_returns_df['returns'] = np.exp(daily_returns['returns']) - 1
        mean_daily_return = daily_returns_df['returns'].mean()
        # 计算年化收益率
        # # 计算累积收益率
        # cumulative_return = np.prod(1 + daily_returns['returns']) - 1
        # # 计算年化收益率
        # annualized_return = (1 + cumulative_return) ** (252 / len(daily_returns)) - 1
        annualized_return =daily_returns_df['returns'].mean()*252

        # 计算年化波动率
        daily_volatility = daily_returns_df['returns'].std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        # 计算夏普率
        sharpe_ratio = (mean_daily_return- risk_free_rate) / daily_volatility
        # 计算下行偏差和索提诺比率
        downside_returns = daily_returns_df[daily_returns_df['returns'] < 0]['returns']
        downside_deviation = downside_returns.std()
        sortino_ratio = (mean_daily_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan

        cumulative_returns = (1 + daily_returns_df['returns']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdowns.min())

        return {
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Maximum Drawdown": max_drawdown
        }
    metrics = calculate_performance_metrics(daily_returns )
    # for key, value in metrics.items():
    #     print(f"{key}: {value:.4f}")
    metrics_df1 = pd.DataFrame([metrics])
    metrics_df = pd.concat([metrics_df,metrics_df1])


    # 绘制累计收益的图
    def plot_cumulative_returns(daily_returns_df):
        if 'returns' not in daily_returns_df.columns:
            raise ValueError("DataFrame must contain a 'returns' column")
        daily_returns_df['cumulative_returns'] = (1 + daily_returns_df['returns']).cumprod() - 1
        plt.figure(figsize=(12, 6))
        plt.plot(daily_returns_df.index, daily_returns_df['cumulative_returns'], label='Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
    daily_returns.index = pd.date_range(start='2023-06-30', periods=len(daily_returns), freq='D')
    # plot_cumulative_returns(daily_returns)
print(metrics_df.iloc[0:7,0:4])
metrics_df.to_csv('performance_metrics_kan_{}.csv'.format(i), index=False)
