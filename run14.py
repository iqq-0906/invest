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
        # Network initialization and evaluation functions
        set_seed(10)
        self.model = model

    def reshape(self, X):
        reshaped_X = X.reshape(-1, )
        return reshaped_X
    def helper(self, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10):
        reshaped_x1 = self.reshape(x1)
        reshaped_x2 = self.reshape(x2)
        reshaped_x3 = self.reshape(x3)
        reshaped_x4 = self.reshape(x4)
        reshaped_x5 = self.reshape(x5)
        reshaped_x6 = self.reshape(x6)
        reshaped_x7 = self.reshape(x7)
        reshaped_x8 = self.reshape(x8)
        reshaped_x9 = self.reshape(x9)
        reshaped_x10 = self.reshape(x10)


        stacked_tensor = torch.stack([reshaped_x1, reshaped_x2, reshaped_x3, reshaped_x4, reshaped_x5,
                                      reshaped_x6, reshaped_x7, reshaped_x8, reshaped_x9, reshaped_x10])
        permuted_tensor = stacked_tensor.permute(1, 0)
        return permuted_tensor
    # Define DeepONet architecture
    def operator_net(self,x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        y=self.helper(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)
        outputs = self.model(y)
        return outputs

    # Define PDE residual
    def residual_net(self,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10):
        s=self.operator_net(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)
        s_x1 =jacrev(self.operator_net,argnums=0)(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10).sum(dim=0)
        # s_x12 = torch.autograd.grad(s,x1,grad_outputs=torch.ones_like(s),retain_graph=True,create_graph=True)[0]
        s_x2 =jacrev(self.operator_net, argnums=1)(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10).sum(dim=0)
        s_x3 =jacrev(self.operator_net, argnums=2)(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10).sum(dim=0)
        s_x4 =jacrev(self.operator_net, argnums=3)(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10).sum(dim=0)
        s_x5 =jacrev(self.operator_net, argnums=4)(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10).sum(dim=0)
        s_x6 = jacrev(self.operator_net, argnums=5)(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10).sum(dim=0)
        s_x7 = jacrev(self.operator_net, argnums=6)(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10).sum(dim=0)
        s_x8 = jacrev(self.operator_net, argnums=7)(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10).sum(dim=0)
        s_x9 = jacrev(self.operator_net, argnums=8)(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10).sum(dim=0)
        s_x10 = jacrev(self.operator_net, argnums=9)(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10).sum(dim=0)

        s_x1[s_x1< 0] = 0
        s_x2[s_x2 < 0] = 0
        s_x3[s_x3 < 0] = 0
        s_x4[s_x4 < 0] = 0
        s_x5[s_x5 < 0] = 0
        s_x6[s_x6< 0] = 0
        s_x7[s_x7 < 0] = 0
        s_x8[s_x8 < 0] = 0
        s_x9[s_x9 < 0] = 0
        s_x10[s_x10 < 0] = 0

        return s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10


    def residual_net1(self,x1, x2, x3, x4,  x5, x6, x7, x8, x9, x10,ones):
        s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10=self.residual_net(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)

        output = s_x1 + s_x2 + s_x3 + s_x4 + s_x5 + s_x6 + s_x7 + s_x8 + s_x9 + s_x10
        res=torch.mean((output.flatten() -ones) ** 2)
        return res

    def residual_net2(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        r = self.operator_net(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        r_mean = -torch.mean(r.flatten())
        return r_mean

    def residual_net3(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        r = self.operator_net(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10 = self.residual_net(x1, x2, x3, x4, x5, x6, x7, x8,
                                                                                        x9, x10)

        s1 = s_x1 * x1 + s_x2 * x2 + s_x3 * x3 + s_x4 * x4 + s_x5 * x5 + s_x6 * x6 + s_x7 * x7 + s_x8 * x8 + s_x9 * x9 + s_x10 * x10
        r_mean = torch.mean((s1.flatten()))
        th = self.residual_net42(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        outputs=1.25*th-r_mean
        return outputs

    # def residual_net3(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, covariance_matrix):
    #     s = self.operator_net(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
    #     s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10 = self.residual_net(x1, x2, x3, x4, x5, x6, x7, x8,
    #                                                                                     x9, x10)
    #     s1 = s_x1 * x1 + s_x2 * x2 + s_x3 * x3 + s_x4 * x4 + s_x5 * x5 + s_x6 * x6 + s_x7 * x7 + s_x8 * x8 + s_x9 * x9 + s_x10 * x10
    #     weights = self.helper(s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10)
    #     portfolio_var = torch.matmul(torch.matmul(weights, covariance_matrix), weights.T)
    #     portfolio_var = torch.diag(portfolio_var)
    #     outputs = -(torch.mean(s.flatten() - 1.25 * portfolio_var))
    #     return outputs

    # def residual_net41(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,covariance_matrix):
    #     s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10=self.residual_net(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
    #     weights=self.helper(s_x1,s_x2,s_x3,s_x4,s_x5,s_x6,s_x7,s_x8,s_x9,s_x10)
    #     portfolio_var =torch.matmul(torch.matmul(weights,covariance_matrix), weights.T)
    #     portfolio_var=torch.diag(portfolio_var)
    #     # outputs = -torch.mean((portfolio_var.flatten()**2))
    #     return outputs
    def residual_net42(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        r = self.operator_net(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        outputs = torch.std((r.flatten()))
        return outputs

    def residual_net4(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        r = self.operator_net(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        r_mean = torch.mean(r.flatten())
        th = self.residual_net42(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        sp = -r_mean / th
        return sp




    def residual_net5(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        s = self.operator_net(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10= self.residual_net(x1,x2,x3,x4,x5,x6,x7,x8,x9, x10)

        s1=s_x1*x1+s_x2*x2+s_x3*x3+s_x4*x4+s_x5*x5+s_x6*x6+s_x7*x7+s_x8*x8+s_x9*x9+s_x10*x10
        outputs =torch.mean((s1.flatten() - s.flatten())**2)
        return outputs

    def calculate_cvar(self, r, alpha=0.05):
        var = torch.quantile(r, 1 - alpha)
        cvar = torch.mean(r[r >= var])
        outputs = torch.mean((cvar.flatten()) ** 2)

        return outputs

    def residual_net_cvar(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, alpha):
        r = self.operator_net(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        cvar = self.calculate_cvar(r,alpha)
        return cvar


    def train(self,x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,ones,covariance_matrix):
        # self.optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, history_size=10, line_search_fn="strong_wolfe",
        #                   tolerance_grad=1e-32, tolerance_change=1e-32)
        self.optimizer= torch.optim.Adam(model.parameters(), lr=0.1)


        pbar = tqdm(range(30), desc='description')
        for _ in pbar:
            self.optimizer.zero_grad()

            # 计算各个损失
            loss1 = self.residual_net1(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, ones)
            loss2 = self.residual_net2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
            loss3 = self.residual_net3(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
            loss4 = self.residual_net4(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
            loss5 = self.residual_net5(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
            # if loss2.item() > 0 :
            #     continue
            #     # 跳过当前 epoch，重新训练

            # 合并损失函数，权重根据实际情况调整
            loss =1000*loss1  +loss3+100*loss5

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            if _ % 1 == 0:
                pbar.set_description("loss1: %.2e | loss3: %.2e| loss5: %.2e" %
                                     (loss1.detach().numpy(), loss3.detach().numpy(),
                                      loss5.detach().numpy()
                                      ))




i=64
metrics_df = pd.DataFrame()
for l in range(1, 6):
    pre_data=pd.read_csv('data/BL_returnBL_rolling{}_{}.csv'.format(l,i))
    df = pd.DataFrame()
    for j in range(0,64,i):
        print('{}-{}-{}'.format(i,l,j))
        if i in [5,10,15,20,30,64]:
            if j != 60:
                pre_data_return=pre_data.iloc[j:j+i,:]
                cov_data=pd.read_csv('data\BL_cov_rolling{}_{}_{}.csv'.format(l,i,j))

                del cov_data['Unnamed: 0']
                covariance_matrix = torch.tensor(cov_data.values).float()
                ones = torch.ones(i)
                tensor_list = []
                for m in range(10):
                    numpy_array = pre_data_return.iloc[:, m].to_numpy(dtype=np.float64)
                    tensor_name = torch.tensor(numpy_array, dtype=torch.float64).reshape(-1, ).float()
                    torch.set_printoptions(precision=6)
                    tensor_list.append(tensor_name)
                kan  = KAN(width=[4,2,1,1], grid=3, k=3, seed=1, device=device)
                model = r_th(kan)
                model.train(tensor_list[0],tensor_list[1],tensor_list[2],tensor_list[3],tensor_list[4],tensor_list[5],
                tensor_list[6],tensor_list[7],tensor_list[8],tensor_list[9],ones,covariance_matrix)
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
                cov_data = pd.read_csv(r'data\BL_cov_rolling{}_{}_{}.csv'.format(l, i,j))
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
                            tensor_list[6], tensor_list[7], tensor_list[8], tensor_list[9], ones, covariance_matrix)
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
                cov_data = pd.read_csv(r'data\BL_cov_rolling{}_{}_{}.csv'.format(l, i, j))
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
                            tensor_list[6], tensor_list[7], tensor_list[8], tensor_list[9], ones, covariance_matrix)
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
                cov_data = pd.read_csv(
                    r'data\BL_cov_rolling{}_{}_{}.csv'.format(l, i, j))
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
                            tensor_list[6], tensor_list[7], tensor_list[8], tensor_list[9], ones, covariance_matrix)
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
    true_data = pd.read_csv(r'data\stock_true_data_rolling{}_{}.csv'.format(g,i)).to_numpy()
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
