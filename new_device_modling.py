import pandas as pd
from attribute_general import AttributeGeneral
from sklearn.metrics import roc_auc_score, classification_report
import warnings

warnings.filterwarnings('ignore')

'''
SELECT  imei
        ,source
        ,package
        ,os
        ,IF(language IN ('zh','en'),'main','not_main') language_group
        ,language
        ,CASE   WHEN language = 'zh' OR country = '中国' THEN '中国'
                WHEN country IN ('新加坡','台湾','香港','日本','韩国') THEN '东亚'
                WHEN country IN ('印度尼西亚','印度','孟加拉国','巴基斯坦',' 泰国','越南','斯里兰卡') THEN '南亚'
                WHEN country IN ('美国','英国','法国','西班牙','俄罗斯联邦','德国') THEN '欧美'
                ELSE '其他国家'
        END country_group
        ,country
        ,first_token_trade_coin_pair
        ,first_token_trade_chain_pair
        ,first_nft_trade_token_title
        ,first_connect_dapp
        ,first_otc_fiat_code
        ,airdrop_name
        ,invite_code
        ,red_packet_code
        ,amt_usd
        ,red_packet_amount
        ,total_trade
        ,version
        ,red_packet_coin
        ,red_packet_name
        ,main_token_hold_usdt
        ,future_total_trade
        ,lately_login_ip
        ,wallet_type
        ,level_1_user_source
        ,level_2_user_source
        ,level_3_user_source
        ,IF(device_create_date = '2024-04-28',1,0) target
        ,device_create_date
FROM    dm_bi_bk_user_root_analysis_stat_df
WHERE   pt = '${bizdate}'
AND     device_create_date IN ('2024-04-27','2024-04-28')
AND     wallet_type <> 100
;
'''

df = pd.read_csv('data/new_device.csv')

features = ['source', 'package', 'os', 'language', 'country','language_group', 'country_group',
            'first_token_trade_coin_pair',
            'first_token_trade_chain_pair', 'first_nft_trade_token_title',
            'first_connect_dapp', 'first_otc_fiat_code', 'airdrop_name',
            'invite_code', 'red_packet_code',
            'amt_usd', 
            'total_trade', 
            'future_total_trade', 'lately_login_ip',
            'wallet_type', 'level_1_user_source',
            'level_2_user_source', 'level_3_user_source']
cat_features = ['source', 'package', 'os', 'language', 'country','language_group', 'country_group',
            'first_token_trade_coin_pair',
            'first_token_trade_chain_pair', 'first_nft_trade_token_title',
            'first_connect_dapp', 'first_otc_fiat_code', 'airdrop_name',
            'invite_code', 'red_packet_code',
            'lately_login_ip',
            'wallet_type', 'level_1_user_source',
            'level_2_user_source', 'level_3_user_source']
target = 'target'

def count_func(query, time_col, time_set, df=df):
    try:
        if query != '':
                return df[df[time_col] == time_set].query(query).shape[0]
        else:
                return df[df[time_col] == time_set].shape[0]
    except:
          print(query)
          raise ValueError


t = AttributeGeneral(df, features, cat_features, target)
t.train()

print('Training classification report: \n', classification_report(t.y_train, t.best_model.predict(t.X_train)))
print('Training auc: \n', roc_auc_score(t.y_train, t.best_model.predict_proba(t.X_train)[:, 1]))

model = t.best_model
tree_df = t.get_model_visualization()

tree_df['2024-04-27'] = tree_df['conditions'].apply(lambda x: count_func(x, 'device_create_date', '2024-04-27'))
tree_df['2024-04-28'] = tree_df['conditions'].apply(lambda x: count_func(x, 'device_create_date', '2024-04-28'))
tree_df['ratio'] = (tree_df['2024-04-28'] - tree_df['2024-04-27'])/(tree_df[tree_df['conditions']=='all']['2024-04-28'].values[0] - tree_df[tree_df['conditions']=='all']['2024-04-27'].values[0])

print('******************AttributeGeneral******************')
tree_df[['node_index', 'split_feature', 'left_child','right_child', '2024-04-27', '2024-04-28', 'ratio', 'condition', 'conditions', ]].to_csv('data/new_device_tree.csv', index=False)

print(tree_df[tree_df['ratio'] > 0][['conditions', '2024-04-27', '2024-04-28', 'ratio']].head(5))

print(list(zip(t.features, model.feature_importances_)))

# print(df.pivot_table(index='target', values='imei', aggfunc='count'))
# print(
#     df[df[tree_df[tree_df['node_index'] == 0]['split_feature'].values[0]].isin(tree_df[tree_df['node_index'] == 0]['thres_vis'].values[0])].pivot_table(index='target', values='imei', aggfunc='count').reset_index()
# )