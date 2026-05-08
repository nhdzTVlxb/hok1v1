from agent_ppo.utils import is_iterable

def dfs_iter_apply_fn(x, fn, only_dict=True, input_key=False, belong_key=None, inplace=False, passby=None, only_leaf=True):
    """ 递归地对字典x的叶子结点作用func函数.
    Args:
        x: 待处理字典
        fn: 作用于叶子节点的函数
        only_dict: 只对字典递归
        input_key: 将从属的key值作为输入, 输入到fn中 (从属于list或为根节点时, belong_key=None)
        only_leaf: 仅在叶子节点上作用fn函数
    """
    inputs = [x]
    if input_key: inputs.append(belong_key)
    if passby is not None: inputs.append(passby)
    if (not is_iterable(x)) or (not isinstance(x, dict) and only_dict):
        return fn(*inputs)
    if not only_leaf:
        fn(*inputs)
    if isinstance(x, dict):
        tmp = {}
        for k, v in x.items():
            tmp[k] = dfs_iter_apply_fn(v, fn, only_dict, input_key, k, inplace, passby, only_leaf)
            if inplace:
                x[k] = tmp[k]
    if isinstance(x, list) and not only_dict:
        tmp = []
        for i, elem in enumerate(x):
            tmp.append(dfs_iter_apply_fn(elem, fn, only_dict, input_key, None, inplace, passby, only_leaf))
            if inplace:
                x[i] = tmp[-1]
    return tmp

if __name__ == '__main__':
    x = {'a': [{'aa': 3}, {'bb': 5},4 ], 'b': {'bb': [1,2], 'd': 4}, 'e': 5}
    # fn = lambda x: -x
    ### DEBUG1 ###
    import numpy as np
    def fn(x):
        if isinstance(x, list):
            tmp = np.array(x)
            if tmp.dtype == object:    # 对于非数值型的list进一步进行迭代
                return [dfs_iter_apply_fn(i, fn) for i in x]
            return f'type={tmp.dtype}, shape={tmp.shape}'
        return x
    y = dfs_iter_apply_fn(x, fn)
    print(x, y, sep='\n')
    ### DEBUG2 ###
    fn = lambda x, key: key + '/belong' if key is not None else 'root/list'
    z = dfs_iter_apply_fn(x, fn, only_dict=False, input_key=True)
    print(z)
    # dfs_iter_apply_fn(x, fn, only_dict=False, input_key=True, inplace=True)
    ### DEBUG3 ###
    def fn(x, key, passby: list):
        if key == 'bb':
            passby.append(x)
    l = []
    dfs_iter_apply_fn(x, fn, only_dict=False, input_key=True, passby=l, only_leaf=False)
    print(l)
    