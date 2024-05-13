import sys
sys.path.append('src')

from TreeAttrClassify import TreeAttrClassify
import pandas as pd
import yaml
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='TreeAttrCls')
parser.add_argument('--config', help='config path', required=True)
args = parser.parse_args()
config = args.config


with open(config, 'r') as f:
    config = yaml.safe_load(f)

print(config['task_name'])

df = pd.read_csv(config['data_path'])

features = config['features']
target = config['target']

t = TreeAttrClassify(df, features, target, n_trials=20)
t.train()

tree_df, best_path, rq, message = t(config['time_col'], config['time_comp'], config['pcol'])
print('\n\n**Task: {}**\n{}\n'.format(config['task_name'], message))