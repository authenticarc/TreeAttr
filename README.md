# TreeAttr
Use Lightgbm to attribute data change

- 活跃
- 新增
- 持仓转化
- swap交易人数和金额
- future交易人数和金额

## swap数据归因

```bash
python3 scripts/reg.py --config config/swap_amt.yaml | grep -v "LightGBM"
python3 scripts/cls.py --config config/swap_user.yaml | grep -v "LightGBM"
```

```sql
SELECT  a.*
        ,b.country
        ,b.language
        ,b.device_create_date
        ,b.imei_level_1_user_source
        ,b.imei_level_2_user_source
        ,b.imei_level_3_user_source
        ,b.lifecycle_tag
        ,if(b.first_token_trade_date = dt, 'new', 'old') new_swap
FROM    (
            SELECT  identity
                    ,CONCAT_WS('-',from_coin,to_coin) coin_pair
                    ,CONCAT_WS('-',from_chain,to_chain) chain_pair
                    ,market
                    ,from_amt_usd
                    ,IF(date(create_time) = DATE_SUB(current_date(),2),0,1) target
                    ,date(create_time) dt
                    ,order_no
                    ,HOUR(create_time) create_hour
            FROM    bg_dw.dwd_bk_trs_swap_order_df
            WHERE   pt = '${bizdate}'
            AND     order_status = 4
            AND     date(create_time) BETWEEN DATE_SUB(current_date(),2) AND DATE_SUB(current_date(),1)
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
                        ,date(first_token_trade_time) first_token_trade_date
                FROM    bg_dw.dws_bk_usr_profile_subset_df
                WHERE   pt = '${bizdate}'
            ) b
ON      a.identity = b.identity
;
```

```yaml
# For classify task

# The name of the task
task_name: Swap amt Change
data_path: data/swap.csv

features:
- coin_pair
- chain_pair 
- market
- country 
- language 
- imei_level_1_user_source 
- imei_level_2_user_source
- imei_level_3_user_source 
- lifecycle_tag
- dt
- new_swap

target: from_amt_usd

time_col: dt
time_comp:
- "2024-05-11"
- "2024-05-12"

pcol: from_amt_usd
```

```yaml
# For classify task

# The name of the task
task_name: Swap Identity Change
data_path: data/swap.csv

features:
- coin_pair
- chain_pair 
- market
- country 
- language 
- from_amt_usd 
- imei_level_1_user_source 
- imei_level_2_user_source
- imei_level_3_user_source 
- lifecycle_tag
- create_hour
- new_swap

target: target

time_col: dt
time_comp:
- "2024-05-11"
- "2024-05-12"

pcol: identity
```

## futures用户归因

```bash
python3 scripts/cls.py --config config/futures_user.yaml | grep -v "LightGBM"
python3 scripts/reg.py --config config/futures_amt.yaml | grep -v "LightGBM"
```

```sql
WITH base AS 
(
    SELECT  user_id
            ,reg_time
    FROM    bg_dw.dwd_usr_basic_info_df
    WHERE   pt = '${bizdate}'
    AND     system_parent_id = '1927119253'
)
SELECT  CONCAT('bg',CAST(a.user_id AS STRING)) user_id
        ,CONCAT('bg',CAST(c.order_id AS STRING)) order_id
        ,c.trade_amt
        ,symbol_id
        ,d.country
        ,d.language
        ,d.device_create_date
        ,d.imei_level_1_user_source
        ,d.imei_level_2_user_source
        ,d.imei_level_3_user_source
        ,d.lifecycle_tag
        ,dt
        ,target
        ,if(d.first_future_trade_date = dt, 1, 0) new_future
FROM    base a
INNER JOIN  (
                SELECT  user_id
                        ,order_id
                        ,trade_amt
                        ,date(trade_time) dt
                        ,symbol_id
                        ,IDENTITY
                        ,IF(date(trade_time) = DATE_SUB(current_date(),2),0,1) target
                FROM    bg_dw.dwd_bk_trs_usr_dealt_record_df
                WHERE   pt = '${bizdate}'
                AND     date(trade_time) BETWEEN DATE_SUB(current_date(),2) AND DATE_SUB(current_date(),1)
            ) c
ON      a.user_id = c.user_id
LEFT JOIN   (
                SELECT  IDENTITY
                        ,country
                        ,language
                        ,date(device_create_time) device_create_date
                        ,imei_level_1_user_source
                        ,imei_level_2_user_source
                        ,imei_level_3_user_source
                        ,lifecycle_tag
                        ,date(first_future_trade_time) first_future_trade_date
                        ,token_trade_success_amt_usd
                        ,token_hold_usdt
                        ,his_usd
                FROM    bg_dw.dws_bk_usr_profile_subset_df
                WHERE   pt = '${bizdate}'
            ) d
ON      c.identity = d.identity
;
```

## 新增设备

```bash
python3 scripts/cls.py --config config/new_imei.yaml | grep -v "LightGBM"
```

```sql
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
        ,IF(device_create_date = DATE_SUB(current_date(),1),1,0) target
        ,device_create_date
FROM    dm_bi_bk_user_root_analysis_stat_df
WHERE   pt = '${bizdate}'
AND     device_create_date between DATE_SUB(current_date(),2) AND DATE_SUB(current_date(),1)
AND     wallet_type <> 100
;
```