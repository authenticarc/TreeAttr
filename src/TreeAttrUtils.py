'''
Author: Clyde Ren
Date: 2024-05-13 01:22:00

定义了一些和TreeAttr相关的工具函数
- tree_to_table: 将树结构转化成表格形式
- combine_conditions: 合并条件
- best_path: 获取最佳路径
- find_path: 查找路径
- get_sign: 获取符号
'''

import pandas as pd
         
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

def best_path(tree_df, depth=1):
    tree_dict = tree_df.set_index('node_index').to_dict('index')

    if depth == 0:
        raise ValueError("Depth must be greater than 0")
    
    max_sum = 0
    best_path = []

    def find_paths(node, current_path, current_ids, current_length):
        if current_length == depth or pd.isna(tree_dict[node]['left_child']) or pd.isna(tree_dict[node]['right_child']):
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

def get_sign(a,b):
    return 1 if a > b else -1 if a < b else 0