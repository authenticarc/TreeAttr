import pandas as pd
from attribute_general import AttributeGeneral
from sklearn.metrics import roc_auc_score, classification_report
import warnings

warnings.filterwarnings('ignore')


'''
SELECT  a.*
        ,b.country
        ,b.language
        ,b.device_create_date
        ,b.imei_level_1_user_source
        ,b.imei_level_2_user_source
        ,b.imei_level_3_user_source
        ,b.lifecycle_tag
FROM    (
            SELECT  identity
                    ,nvl(from_coin, from_contract) from_coin
                    ,nvl(to_coin, to_contract) to_coin
                    ,from_chain
                    ,to_chain
                    ,market
                    ,from_amt_usd
                    ,IF(date(create_time) = '2024-05-03',0,1) target
                    ,date(create_time) dt
            FROM    bg_dw.dwd_bk_trs_swap_order_df
            WHERE   pt = '${bizdate}'
            AND     order_status = 4
            AND     date(create_time) BETWEEN '2024-05-03' AND '2024-05-04'
        ) a
LEFT JOIN   (
                SELECT  identity
                        ,country
                        ,language
                        ,date(device_create_time) device_create_date
                        ,imei_level_1_user_source
                        ,imei_level_2_user_source
                        ,imei_level_3_user_source
                        ,lifecycle_tag
                FROM    bg_dw.dws_bk_usr_profile_subset_df
                WHERE   pt = '${bizdate}'
            ) b
ON      a.identity = b.identity
;
'''

df = pd.read_csv('data/swap.csv')

features = ['from_coin', 'to_coin', 'from_chain', 'to_chain', 'market',
            'country', 'language', 
            'imei_level_1_user_source', 'imei_level_2_user_source',
            'imei_level_3_user_source', 'lifecycle_tag', 'from_amt_usd']
cat_features = ['from_coin', 'to_coin', 'from_chain', 'to_chain', 'market',
            'country', 'language', 
            'imei_level_1_user_source', 'imei_level_2_user_source',
            'imei_level_3_user_source', 'lifecycle_tag']
target = 'target'

def count_func(query, time_col, time_set, df=df):
    if query != '':
        return df[df[time_col] == time_set].query(query).shape[0]
    else:
        return df[df[time_col] == time_set].shape[0]


t = AttributeGeneral(df, features, cat_features, target)
t.train()

print('Training classification report: \n', classification_report(t.y_train, t.best_model.predict(t.X_train)))
print('Training auc: \n', roc_auc_score(t.y_train, t.best_model.predict_proba(t.X_train)[:, 1]))

model = t.best_model
tree_df = t.get_model_visualization()

tree_df['2024-05-03'] = tree_df['conditions'].apply(lambda x: count_func(x, 'dt', '2024-05-03'))
tree_df['2024-05-04'] = tree_df['conditions'].apply(lambda x: count_func(x, 'dt', '2024-05-04'))
tree_df['ratio'] = (tree_df['2024-05-04'] - tree_df['2024-05-03'])/(tree_df[tree_df['conditions']=='']['2024-05-04'].values[0] - tree_df[tree_df['conditions']=='']['2024-05-03'].values[0])

print('******************AttributeGeneral******************')
tree_df[['node_index', 'split_feature', 'left_child','right_child', '2024-05-03', '2024-05-04', 'ratio', 'condition', 'conditions', ]].to_csv('data/swap_tree.csv', index=False)

print(tree_df[tree_df['ratio'] > 0][['conditions', '2024-05-03', '2024-05-04', 'ratio']].head(5))

print(list(zip(t.features, model.feature_importances_)))

# print(df.pivot_table(index='target', values='imei', aggfunc='count'))
# print(
#     df[df[tree_df[tree_df['node_index'] == 0]['split_feature'].values[0]].isin(tree_df[tree_df['node_index'] == 0]['thres_vis'].values[0])].pivot_table(index='target', values='imei', aggfunc='count').reset_index()
# )