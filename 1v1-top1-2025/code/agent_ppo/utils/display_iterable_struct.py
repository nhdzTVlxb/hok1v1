"""
用于对嵌套的可迭代对象（如list和dict）进行递归处理，简化输出格式。
存在两种主要函数：
1. `simplify_iter`: 对所有叶子结点为list的进行简化，输出type和shape信息。
2. `too_simplify_iter`: 对所有叶子结点为list的进行简化，输出type和shape信息，但不对object对象的list进行展开。
"""

import numpy as np
import json

def is_iterable(x):
    return isinstance(x, list) or isinstance(x, dict)

def dfs_iter_apply_fn(x, fn, only_dict=True):
    """ 递归地对x的叶子结点作用func函数, 主要对dict元素进行展开, 对list可以作用fn函数进行缩写
    相关函数:
        - simplify_iter
        - too_simplify_iter
    """
    if (not is_iterable(x)) or (not isinstance(x, dict) and only_dict):
        return fn(x)
    if isinstance(x, dict):
        tmp = {}
        for k, v in x.items():
            tmp[k] = dfs_iter_apply_fn(v, fn, only_dict=only_dict)
    if isinstance(x, list) and not only_dict:
        tmp = []
        for i in x:
            tmp.append(dfs_iter_apply_fn(i, fn, only_dict=only_dict))
    return tmp

def save_json(x, save_path):
    """ 将x保存为JSON文件 """
    with open(save_path, 'w') as file:
        json.dump(x, file, indent=2)

def too_simplify_iter(x, save_path=None):
    """ 对x的所有叶子结点为list的进行简化, 输出type和shape信息, 
    但不对object对象的list（list中同时存在数值, list, dict时ndarray自动转为object类型）再进行展开
    Args:
        x: 需要简化的对象
        save_path: 如果提供了路径，则将简化后的结果保存为JSON文件
    Returns:
        x: 简化后的对象
    """
    def fn(x):
        if isinstance(x, list):
            tmp = np.array(x)
            return f'type={tmp.dtype}, shape={tmp.shape}'
        return x
    x = dfs_iter_apply_fn(x, fn)
    if save_path is not None:
        save_json(x, save_path)
    return x

def simplify_iter(x, save_path=None):
    """ 对x的所有叶子结点为list的进行简化, 输出type和shape信息, 对全部对象进行展开
    Args:
        x: 需要简化的对象
        save_path: 如果提供了路径，则将简化后的结果保存为JSON文件
    Returns:
        x: 简化后的对象
    """
    def fn(x):
        if isinstance(x, list):
            tmp = np.array(x)
            if tmp.dtype == object:
                return [dfs_iter_apply_fn(i, fn) for i in x]
            return f'type={tmp.dtype}, shape={tmp.shape}'
        return x
    x = dfs_iter_apply_fn(x, fn)
    if save_path is not None:
        save_json(x, save_path)
    return x

if __name__ == '__main__':
    from pathlib import Path
    path_log_dir = Path('./agent_ppo/debug')
    print(path_log_dir)
    path_log_dir.mkdir(parents=True, exist_ok=True)
    x = {'a': [{'aa': 3}, {'bb': 5}, 4], 'b': {'c': [1,2], 'd': 4}, 'e': 5, 'f': [1, [1, 2], {'ff': 3}]}
    save_json(x, path_log_dir / 'origin.json')
    y = simplify_iter(x, path_log_dir / 'simplify_iter.json')
    z = too_simplify_iter(x, path_log_dir / 'too_simplify_iter.json')
    print(x, y, z, sep='\n')

"""
{'a': [{'aa': 3}, {'bb': 5}, 4], 'b': {'c': [1, 2], 'd': 4}, 'e': 5, 'f': [1, [1, 2], {'ff': 3}]}
{'a': [{'aa': 3}, {'bb': 5}, 4], 'b': {'c': 'type=int64, shape=(2,)', 'd': 4}, 'e': 5, 'f': [1, 'type=int64, shape=(2,)', {'ff': 3}]}
{'a': 'type=object, shape=(3,)', 'b': {'c': 'type=int64, shape=(2,)', 'd': 4}, 'e': 5, 'f': 'type=object, shape=(3,)'}
"""
