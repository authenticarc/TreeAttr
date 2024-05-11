
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import precision_recall_curve, auc, r2_score
import re


class AttributeGeneral:
    def __init__(self, df: pd.DataFrame, features: list[str], target: str, n_trials: int=20, task_type: str='classification'):
        '''
        df: pd.DataFrame
            数据集
        features: list[str]
            特征列
        target: str
            目标列
        n_trials: int
            超参数搜索次数
        task_type: str (classification, regression)
            任务类型，分类或回归
            例子:
            分类任务: DAU、交易人数等
            回归任务: 交易金额等
        '''
        self.df = df
        self.n_trials = n_trials
        self.task_type = task_type

        if self.task_type not in ['classification', 'regression']:
            raise ValueError('task_type must be classification or regression')
        
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

        if self.task_type == 'classification':
            self.df[self.target] = self.df[self.target].astype('int')
        elif self.task_type == 'regression':
            self.df[self.target] = self.df[self.target].astype('float')

        self.df[self.cat_features] = self.df[self.cat_features].astype('category')
    
    def __objective__(self, trial):
        '''
        优化目标
        '''
        if self.task_type == 'classification':
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'goss', 'dart']),
                'max_depth': trial.suggest_int('max_depth', 6, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256)
            }
            
            # 使用Lightgbm创建模型并设置样本权重
            model = lgb.LGBMClassifier(**params, n_jobs=-1, categorical_feature=self.cat_features, random_state=42)
            model.fit(self.X_train, self.y_train)

            # 验证模型性能pr-auc
            precision, recall, _ = precision_recall_curve(self.y_train, model.predict_proba(self.X_train)[:, 1])
            
            return -auc(recall, precision)
        elif self.task_type == 'regression':
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'goss', 'dart']),
                'max_depth': trial.suggest_int('max_depth', 6, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256)
            }
            
            # 使用Lightgbm创建模型并设置样本权重
            model = lgb.LGBMRegressor(**params, n_jobs=-1, categorical_feature=self.cat_features, random_state=42)
            model.fit(self.X_train, self.y_train)

            r2 = r2_score(self.y_train, model.predict(self.X_train))

            return -r2
        else:
            raise ValueError('Task type not supported')
        
    def train(self):
        '''
        训练模型
        '''
        
        self.X_train, self.y_train = self.df[self.features], self.df[self.target]

        study = optuna.create_study(direction='minimize')
        study.optimize(self.__objective__, n_trials=self.n_trials, n_jobs=-1, timeout=600)

        best_params = study.best_params
        print("Best Hyperparameters: ", best_params)

        if self.task_type == 'classification':
            self.best_model = lgb.LGBMClassifier(**best_params)
            self.best_model.fit(self.X_train, self.y_train)
        elif self.task_type == 'regression':
            self.best_model = lgb.LGBMRegressor(**best_params)
            self.best_model.fit(self.X_train, self.y_train)
        else:
            raise ValueError('Task type not supported')

        self.d = self.best_model.booster_.dump_model()

        return self.best_model
    
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
        rq = {}

        sort_func = {
            'classification': lambda x, key: (self.df[self.df[key] == x][self.df[time_col] == t2][pcol].nunique() - self.df[self.df[key] == x][self.df[time_col] == t1][pcol].nunique())/(self.df[self.df[time_col] == t2][pcol].nunique() - self.df[self.df[time_col] == t1][pcol].nunique()),
            'regression': lambda x, key: (self.df[self.df[key] == x][self.df[time_col] == t2][pcol].sum() - self.df[self.df[key] == x][self.df[time_col] == t1][pcol].sum())/(self.df[self.df[time_col] == t2][pcol].sum() - self.df[self.df[time_col] == t1][pcol].sum())
        }[self.task_type]

        for q in query:
            if q.startswith('~'):
                pattern = r'([a-zA-Z_]+) in '
                match = re.search(pattern, q)
                key = match.group(1)
                value_a = eval(q.split(' in ')[-1].strip(')').strip('('))
                value_b = self.df[~self.df[key].isin(value_a)][key].unique().tolist()
                if key in rq.keys():
                    value = list(set(rq[key]).union(set(value_b)))
                    value = sorted(value_b, key=lambda x: sort_func(x, key = key), reverse=True)
                    rq[key] = value
                else:
                    value = sorted(value_b, key=lambda x: sort_func(x, key = key), reverse=True)
                    rq[key] = list(set(value))
            elif 'in' in q:
                key = q.split(' in ')[0]
                value = eval(q.split(' in ')[-1].strip(')').strip('('))
                if key in rq.keys():
                    value = list(set(rq[key]).union(set(value)))
                    value = sorted(value, key=lambda x: sort_func(x, key = key), reverse=True)
                    rq[key] = value
                else:
                    value = sorted(value, key=lambda x: sort_func(x, key = key), reverse=True)
                    rq[key] = list(set(value))
            elif '<=' in q:
                key = q.split(' <= ')[0]
                value = eval(q.split(' <= ')[-1].strip(')').strip('('))
                if key in rq.keys():
                    if value < rq[key]:
                        rq[key] = value
                    else:
                        pass

        return rq
            
    
    @staticmethod
    def tree_to_table(tree, feature_names):
        # 初始化一个空的DataFrame
        df = pd.DataFrame(columns=["node_index", "split_feature_indx", "split_feature", "threshold", "decision_type", "left_child", "right_child", "leaf_value"])

        def parse_node(node, node_index):
            nonlocal df
            if 'split_feature' in node:
                # 如果节点是内部节点，添加分裂特征、阈值和决策类型
                df = df.append({"node_index": node_index, 
                                "split_feature_indx": node["split_feature"], 
                                "split_feature": feature_names[node["split_feature"]], 
                                "threshold": node["threshold"], 
                                "decision_type": node["decision_type"], 
                                "left_child": node_index * 2 + 1, 
                                "right_child": node_index * 2 + 2, 
                                "leaf_value": None,
                                }, ignore_index=True)
                # 递归处理子节点
                parse_node(node['left_child'], node_index * 2 + 1)
                parse_node(node['right_child'], node_index * 2 + 2)
            else:
                # 如果节点是叶子节点，添加叶子值
                df = df.append({"node_index": node_index, 
                                "split_feature_indx":None,
                                "split_feature": None, 
                                "threshold": None, 
                                "decision_type": None, 
                                "left_child": None, 
                                "right_child": None, 
                                "leaf_value": node["leaf_value"]}, ignore_index=True)

        # 从根节点开始解析
        parse_node(tree, 0)
        
        return df
        
    @staticmethod
    def combine_conditions(tree_df):
        # 将dataframe转化成dict，提高运行效率
        tree_dict = tree_df.set_index('node_index').to_dict('index')

        collector = {}

        # 递归函数，用于收集每个节点的条件
        def collect(node, conditions):
            nonlocal collector
            node_data = tree_dict[node]
            current_condition = node_data['condition']

            collector[node] = conditions.lstrip(' & ')

            left_child = node_data['left_child']
            right_child = node_data['right_child']

            if pd.isna(left_child) and pd.isna(right_child):
                return 

            if not pd.isna(left_child):
                collect(left_child, f"{conditions} & {current_condition}")
            if not pd.isna(right_child):
                collect(right_child, f"{conditions} & ~({current_condition})")

        collect(0, '')
        return collector
    
    @staticmethod
    def best_path(tree_df, depth=1):
        tree_dict = tree_df.set_index('node_index').to_dict('index')

        if depth == 0:
            raise ValueError("Depth must be greater than 0")
        
        max_sum = 0
        best_path = []

        def find_paths(node, current_path, current_ids, current_length):
            if current_length == depth:
                nonlocal max_sum, best_path
                current_sum = sum(current_path)
                if current_sum > max_sum:
                    max_sum = current_sum
                    best_path = current_ids.copy()
                return
            
            for child in ['left_child', 'right_child']:
                child_id = tree_dict[node][child]
                if pd.isna(child_id):
                    return
                child_value = tree_dict[child_id]['ratio']
                if child_value > 0:
                    find_paths(child_id, current_path + [child_value], current_ids + [child_id], current_length + 1)
                else:
                    continue

        find_paths(0, [], [0], 1)
        return max_sum, best_path
    
    @staticmethod
    def find_path(tree_df, target, max_depth=5):
        tree_dict = tree_df.set_index('node_index').to_dict(orient='index')
        paths = []

        def dfs(node_index, current_path, current_depth):
            if current_depth > max_depth:
                return
            
            current_path.append(node_index)

            if pd.isna(tree_dict[node_index]['condition']):
                return

            if target == tree_dict[node_index]['split_feature']:
                paths.append(list(current_path))
                return

            if pd.isna(tree_dict[node_index].get('left_child')) or pd.isna(tree_dict[node_index].get('right_child')):
                return

            dfs(tree_dict[node_index]['left_child'], current_path.copy(), current_depth + 1)
            dfs(tree_dict[node_index]['right_child'], current_path.copy(), current_depth + 1)
            current_path.pop()
            
        dfs(0, [], 0)
        paths = sorted(paths, key=lambda x: len(x))
        return paths
    
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

        self.tree_df = self.tree_to_table(self.d['tree_info'][0]['tree_structure'], self.d['feature_names']) # 生成树结构表
        self.tree_df.sort_values('node_index', inplace=True)   # 按照节点索引排序
        self.tree_df['thres_vis'] = self.tree_df.apply(self.clean_threshold, axis=1) # 清洗阈值
        self.tree_df['condition'] = self.tree_df.apply(self.clean_condition, axis=1) # 清洗条件
        conditions = pd.DataFrame(list(self.combine_conditions(self.tree_df).items()), columns=['node_index', 'conditions']) # 生成条件表
        self.tree_df = pd.merge(self.tree_df, conditions, how='left', on='node_index') # 合并条件表

        if self.task_type == 'classification':
            self.tree_df[self.comp_time[0]] = self.tree_df['conditions'].apply(lambda x: self.count_func(x, self.time_col, self.comp_time[0], pcol))
            self.tree_df[self.comp_time[1]] = self.tree_df['conditions'].apply(lambda x: self.count_func(x, self.time_col, self.comp_time[1], pcol))
            self.tree_df['ratio'] = (self.tree_df[self.comp_time[1]] - self.tree_df[self.comp_time[0]])/(self.tree_df[self.tree_df['conditions']==''][self.comp_time[1]].values[0] - self.tree_df[self.tree_df['conditions']==''][self.comp_time[0]].values[0])

            _, best_path = self.best_path(self.tree_df, depth)

            rq = self.get_rq(best_path, time_col, self.comp_time[0], self.comp_time[1], pcol)
            
        elif self.task_type == 'regression':
            self.tree_df[self.comp_time[0]] = self.tree_df['conditions'].apply(lambda x: self.sum_func(x, self.time_col, self.comp_time[0], pcol))
            self.tree_df[self.comp_time[1]] = self.tree_df['conditions'].apply(lambda x: self.sum_func(x, self.time_col, self.comp_time[1], pcol))
            self.tree_df['ratio'] = (self.tree_df[self.comp_time[1]] - self.tree_df[self.comp_time[0]])/(self.tree_df[self.tree_df['conditions']==''][self.comp_time[1]].values[0] - self.tree_df[self.tree_df['conditions']==''][self.comp_time[0]].values[0])

            max_node_index = self.tree_df[self.tree_df['split_feature'] == 'dt'].sort_values('ratio', ascending=False)['node_index'].values[0]
            
            paths = self.find_path(self.tree_df, target='dt', max_depth=depth)
            for path in paths:
                if path[-1] == max_node_index:
                    best_path = path
                    break
                else:
                    continue

            rq = self.get_rq(best_path, time_col, self.comp_time[0], self.comp_time[1], pcol)

        
        message = ""
        message += f"昨日总体{self.comp_time[0]}: {self.tree_df[self.tree_df['node_index']==0][self.comp_time[0]].values[0]}\n"
        message += f"今日总体{self.comp_time[1]}: {self.tree_df[self.tree_df['node_index']==0][self.comp_time[1]].values[0]}\n"
        message += f"归因路径\n"

        for key, value in rq.items():
            if type(value) is list:
                message += f"{key}: {', '.join(value[:10])}等\n"
            else:
                message += f"{key}: <= {value}\n"

        message += f"昨日{self.comp_time[0]}: {self.tree_df[self.tree_df['node_index']==best_path[-1]][self.comp_time[0]].values[0]}\n"
        message += f"今日{self.comp_time[1]}: {self.tree_df[self.tree_df['node_index']==best_path[-1]][self.comp_time[1]].values[0]}\n"
        message += f"贡献度: {self.tree_df[self.tree_df['node_index']==best_path[-1]]['ratio'].values[0]:.2f}\n"


        return self.tree_df, best_path, rq, message