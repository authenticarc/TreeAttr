from TreeAttrBase import TreeAttrBase
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import auc, precision_recall_curve
from TreeAttrUtils import *
from datetime import datetime


class TreeAttrClassify(TreeAttrBase):
    def __init__(self, df: pd.DataFrame, features: list[str], target: str, n_trials: int=20):
        '''
        df: pd.DataFrame
            数据集
        features: list[str]
            特征列
        target: str
            目标列
        n_trials: int
            超参数搜索次数
        '''
        self.df = df
        self.n_trials = n_trials
        
        self.__preprocessing__(features, target)
        

    def __preprocessing__(self, features: list[str], target: str):
        '''
        特征预处理
        '''
        self.features = features
        self.cat_features = []
        self.target = target
        
        for col in self.features:
            if self.df[col].dtypes == 'object' or self.df[col].dtypes == 'category':
                self.cat_features.append(col)

        self.df[self.target] = self.df[self.target].astype('int')

        self.df[self.cat_features] = self.df[self.cat_features].astype('category')
    
    def __objective__(self, trial):
        '''
        优化目标
        '''
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'goss', 'dart']),
            'max_depth': trial.suggest_int('max_depth', 6, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'verbose': -1
        }
        
        # 使用Lightgbm创建模型并设置样本权重
        model = lgb.LGBMClassifier(**params, n_jobs=-1, categorical_feature=self.cat_features, random_state=42)
        model.fit(self.X_train, self.y_train)

        # 验证模型性能pr-auc
        precision, recall, _ = precision_recall_curve(self.y_train, model.predict_proba(self.X_train)[:, 1])
        
        return -auc(recall, precision)
        
    def train(self):
        '''
        训练模型
        '''
        
        self.X_train, self.y_train = self.df[self.features], self.df[self.target]

        study = optuna.create_study(direction='minimize')
        study.optimize(self.__objective__, n_trials=self.n_trials, n_jobs=-1, timeout=600)

        best_params = study.best_params
        print("Best Hyperparameters: ", best_params)

        self.best_model = lgb.LGBMClassifier(**best_params)
        self.best_model.fit(self.X_train, self.y_train)

        self.d = self.best_model.booster_.dump_model()

        return self.best_model
    
    def sort_func(self, x, key, time_col, t1, t2, pcol):
        return (self.df[self.df[key] == x][self.df[time_col] == t2][pcol].nunique() - self.df[self.df[key] == x][self.df[time_col] == t1][pcol].nunique())/get_sign(self.df[self.df[time_col] == t2][pcol].nunique(), self.df[self.df[time_col] == t1][pcol].nunique())
    
    def __call__(self, time_col: str, comp_time: list, pcol: str = None, depth: int=5):
        '''
        调用函数，生成解释性的决策树表格
        :param time_col: 时间列
        :param comp_time: 时间列的比较条件
        :param pcol: 预测列
        :param depth: 最大深度
        '''
        self.time_col = time_col
        self.comp_time = comp_time

        if pcol is None:
            pcol = self.target

        self.tree_df = tree_to_table(self.d['tree_info'][0]['tree_structure'], self.d['feature_names']) # 生成树结构表
        self.tree_df.sort_values('node_index', inplace=True)   # 按照节点索引排序
        self.tree_df['thres_vis'] = self.tree_df.apply(self.clean_threshold, axis=1) # 清洗阈值
        self.tree_df['condition'] = self.tree_df.apply(self.clean_condition, axis=1) # 清洗条件
        conditions = pd.DataFrame(list(combine_conditions(self.tree_df).items()), columns=['node_index', 'conditions']) # 生成条件表
        self.tree_df = pd.merge(self.tree_df, conditions, how='left', on='node_index') # 合并条件表

        self.tree_df[self.comp_time[0]] = self.tree_df['conditions'].apply(lambda x: self.count_func(x, self.time_col, self.comp_time[0], pcol))
        self.tree_df[self.comp_time[1]] = self.tree_df['conditions'].apply(lambda x: self.count_func(x, self.time_col, self.comp_time[1], pcol))
        self.tree_df['ratio'] = (self.tree_df[self.comp_time[1]] - self.tree_df[self.comp_time[0]])/(self.tree_df[self.tree_df['conditions']==''][self.comp_time[1]].values[0] - self.tree_df[self.tree_df['conditions']==''][self.comp_time[0]].values[0])
        self.tree_df.to_csv('logs/cls-{}.csv'.format(datetime.now()), index=False)

        _, bpath = best_path(self.tree_df, depth)

        rq = self.get_rq(bpath, time_col, self.comp_time[0], self.comp_time[1], pcol)
        message = self.get_message(rq, bpath)

        return self.tree_df, bpath, rq, message
        
