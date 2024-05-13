from abc import ABC, abstractmethod
import pandas as pd
import optuna
import re
from collections import OrderedDict

class TreeAttrBase(ABC):
    def __init__(self, df: pd.DataFrame, features: list[str], target: str, n_trials: int=20):
        self.df = df
        self.n_trials = n_trials
        
        self.__preprocessing__(features, target)
    
    @abstractmethod
    def __preprocessing__(self, features: list[str], target: str):
        pass

    @abstractmethod
    def __objective__(self, trial: optuna.Trial):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def sort_func(self, x, key):
        pass

    def clean_threshold(self, row):
        '''
        清洗阈值
        '''

        try:
            return [self.d['pandas_categorical'][self.cat_features.index(row['split_feature'])][int(i)] for i in row['threshold'].split('||')]
        except:
            return None
        
    def clean_condition(self, row):
        '''
        清洗条件，返回可读的条件
        '''
        try:
            if str(row['decision_type']) == '<=': # 数值型特征一般是返回这一个条件
                return '{} <= {}'.format(row['split_feature'], row['threshold'])
            elif str(row['decision_type']) == '==': # 非数值型特征返回这一条件
                return '{} in {}'.format(row['split_feature'], row['thres_vis']).replace("'", '"')
            else:
                return None
        except:
            return None
        
    def count_func(self, query, time_col, time_set, pcol):
        if query != '':
            return self.df[self.df[time_col] == time_set].query(query)[pcol].nunique()
        else:
            return self.df[self.df[time_col] == time_set][pcol].nunique()
        
    def sum_func(self, query, time_col, time_set, pcol):
        if query != '':
            return self.df[self.df[time_col] == time_set].query(query)[pcol].sum()
        else:
            return self.df[self.df[time_col] == time_set][pcol].sum()
        
    def get_rq(self, best_path, time_col, t1, t2, pcol):
        '''
        获取路径上的查询条件
        '''
        query = self.tree_df[self.tree_df['node_index'] == best_path[-1]]['conditions'].values[0].split(' & ')
        rq = OrderedDict()

        for q in query:
            if 'in' in q:
                if q.startswith('~'):
                    pattern = r'([a-zA-Z_]+) in '
                    match = re.search(pattern, q)
                    key = match.group(1)
                    value_a = eval(q.split(' in ')[-1].strip(')').strip('('))
                    value_b = self.df[~self.df[key].isin(value_a)][key].unique().tolist()
                    value = {i: self.sort_func(i, key=key, time_col=time_col,t1=t1,t2=t2,pcol=pcol) for i in value_b}
                    rq[key] = value
                else:
                    key = q.split(' in ')[0]
                    value = eval(q.split(' in ')[-1].strip(')').strip('('))
                    value = {i: self.sort_func(i, key=key, time_col=time_col,t1=t1,t2=t2,pcol=pcol) for i in value}
                    rq[key] = value
            elif '<=' in q:
                if  q.startswith('~'):
                    key = q.split(' <= ')[0].strip('~(')
                    value = eval(q.split(' <= ')[-1].strip(')').strip('('))
                    if key in rq.keys():
                        if value > rq[key]:
                            rq[key] = {'over_threshold': value}
                    else:
                        rq[key] = {'over_threshold': value}
                else:
                    key = q.split(' <= ')[0]
                    value = eval(q.split(' <= ')[-1].strip(')').strip('('))
                    if key in rq.keys():
                        if value < rq[key]:
                            rq[key] = {'lower_threshold': value}
                    else:
                        rq[key] = {'lower_threshold': value}

        return rq
    
    def get_message(self, rq, best_path):
        message = ""
        message += f"昨日总体{self.comp_time[0]}: {self.tree_df[self.tree_df['node_index']==0][self.comp_time[0]].values[0]}\n"
        message += f"今日总体{self.comp_time[1]}: {self.tree_df[self.tree_df['node_index']==0][self.comp_time[1]].values[0]}\n"
        message += f"归因路径\n"

        for key, value in rq.items():
            if 'over_threshold' in value.keys():
                message += f"{key}: > {value['over_threshold']}\n"
            elif 'lower_threshold' in value.keys():
                message += f"{key}: <= {value['lower_threshold']}\n"
            else:
                message += f"{key}: {', '.join([k for k, _ in sorted(value.items(), key=lambda item:item[1], reverse=True)[:10]])}等\n"

        message += f"昨日{self.comp_time[0]}: {self.tree_df[self.tree_df['node_index']==best_path[-1]][self.comp_time[0]].values[0]}\n"
        message += f"今日{self.comp_time[1]}: {self.tree_df[self.tree_df['node_index']==best_path[-1]][self.comp_time[1]].values[0]}\n"
        message += f"贡献度: {self.tree_df[self.tree_df['node_index']==best_path[-1]]['ratio'].values[0]:.2f}\n"
        return message
   