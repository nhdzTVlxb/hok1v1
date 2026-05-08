import numpy as np
from pprint import pprint

def is_iterable(x):
  return isinstance(x, list) or isinstance(x, dict)

def show_iter(x, indent=2, max_width=50):
    string = ""
    indent_str = ' ' * indent
    if is_iterable(x) and len(str(x)) + indent + 2 > max_width:
        if isinstance(x, dict):
            string += '{\n'
            for k, v in x.items():
                string += indent_str + f"{k}: "
                string += show_iter(v, indent=indent+2)
                string += ',\n'
            string += indent_str[:-2] + '}'
        if isinstance(x, list):
            string += '[\n'
            for v in x:
                string += indent_str + show_iter(v, indent=indent+2) + ',\n'
            string += indent_str[:-2] + ']'
    elif isinstance(x, np.ndarray):
        string += f"({x.dtype}) "
        if x.ndim > 1: string += '\n'
        string += str(x)
    else:
        string += str(x)
    return string

def get_dist(x, y):
  return np.sum((x - y) ** 2) ** 0.5

if __name__ == '__main__':
  a = {'1': [1,2,3], '2': [{'b': 3}, {'a': 2}], '3': np.arange(10).reshape(2, 5), '4': np.arange(3)}
  # a = {'1': [1,2,3], '2': [{'b': 3}, {'a': 2}], }
  print(show_iter(a))
  pprint(a)
