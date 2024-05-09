
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import precision_recall_curve, auc


class AttributeGeneral:
    def __init__(self, df, features, cat_features, target):
        # table = o.get_table(table_name).get_partition('pt={}'.format((datetime.now() - timedelta(days=1)).strftime('%Y%m%d')))
        # self.df = table.to_df().to_pandas()
        # self.df = pd.read_csv(table_name)
        self.df = df
        self._preprocessing(features, cat_features, target)

    def _preprocessing(self, features, cat_features, target):
        self.features = features
        self.cat_features = cat_features
        self.target = target

        self.df[self.cat_features] = self.df[self.cat_features].astype('category')
    
    def _objective(self, trial):
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

        # 验证模型性能
        precision, recall, _ = precision_recall_curve(self.y_train, model.predict_proba(self.X_train)[:, 1])
        
        # return -roc_auc  # Optuna minimizes the objective function, so we use 1 - AUC to maximize AUC
        return -auc(recall, precision)
        
    def train(self):
        
        self.X_train, self.y_train = self.df[self.features], self.df[self.target]

        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective, n_trials=20, n_jobs=-1, timeout=600)  # 设置适当的 n_trials

        best_params = study.best_params
        print("Best Hyperparameters: ", best_params)

        self.best_model = lgb.LGBMClassifier(**best_params)
        self.best_model.fit(self.X_train, self.y_train)
        self.d = self.best_model.booster_.dump_model()

        return self.best_model
    
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
    
    def clean_threshold(self, row):

        try:
            return [self.d['pandas_categorical'][self.cat_features.index(row['split_feature'])][int(i)] for i in row['threshold'].split('||')]
        except:
            return None
        
    def clean_condition(self, row):
        try:
            if str(row['decision_type']) == '<=': # 数值型特征一般是返回这一个条件
                return '{} <= {}'.format(row['split_feature'], row['threshold'])
            elif str(row['decision_type']) == '==': # 非数值型特征返回这一条件
                return '{} in {}'.format(row['split_feature'], row['thres_vis']).replace("'", '"')
            else:
                return None
        except:
            return None
        
    @staticmethod
    def combine_conditions(tree_df):
        # Convert DataFrame to a dictionary for efficient access
        tree_dict = tree_df.set_index('node_index').to_dict('index')

        collector = {}

        def collect(node, conditions):
            node_data = tree_dict[node]
            current_condition = node_data['condition']

            collector[node] = conditions.lstrip(' & ')

            left_child = node_data['left_child']
            right_child = node_data['right_child']

            # Check if children are NaN (pandas.isna) before recursion
            if pd.isna(left_child) and pd.isna(right_child):
                return  # This is a leaf node, stop recursion

            if not pd.isna(left_child):
                collect(left_child, f"{conditions} & {current_condition}")
            if not pd.isna(right_child):
                collect(right_child, f"{conditions} & ~({current_condition})")

        # Start recursive collection from the root node, typically node_index 0
        collect(0, '')
        return collector
            
    def get_model_visualization(self):
        tree_df = self.tree_to_table(self.d['tree_info'][0]['tree_structure'], self.d['feature_names'])
        tree_df.sort_values('node_index', inplace=True)
        tree_df['thres_vis'] = tree_df.apply(self.clean_threshold, axis=1)
        tree_df['condition'] = tree_df.apply(self.clean_condition, axis=1)
        conditions = pd.DataFrame(list(self.combine_conditions(tree_df).items()), columns=['node_index', 'conditions'])
        tree_df = pd.merge(tree_df, conditions, how='left', on='node_index')


        return tree_df