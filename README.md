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


## 新增设备

```bash
python3 scripts/cls.py --config config/new_imei.yaml | grep -v "LightGBM"
```
