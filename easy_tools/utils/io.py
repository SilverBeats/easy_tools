import json
import os
from typing import List, Any, Union, Iterable, Callable
from openpyxl.reader.excel import load_workbook
import pickle
import shutil
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import csv
from .errors import FileTypeError, FileReadError, FileWriteError

__all__ = [
    'read_json',
    'dump_json',
    'read_jsonl',
    'dump_jsonl',
    'read_txt',
    'dump_txt',
    'read_config',
    'dump_config',
    'read_pkl',
    'dump_pkl',
    'read_large_excel',
    'read_excel',
    'dump_excel',
    'read_csv',
    'dump_csv',
    'rm_dir',
    'rm_file',
    'clean_dir',
    'get_file_ext',
    'FileReader',
    'FileWriter'
]


def read_json(file_path: str) -> Any:
    """
    Args:
        file_path: json file path
    """
    if not file_path.endswith('.json'):
        raise FileTypeError(f'{file_path} is not a json file!')
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def dump_json(data: Any, file_path: str, json_encoder=None):
    """
    Args:
        file_path: json file path
        data: data to dump
        json_encoder: for custom situation
    """
    if not file_path.endswith('.json'):
        raise FileTypeError(f'{file_path} is not a json file!')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, default=json_encoder)


def read_jsonl(file_path: str, return_iter: bool = False) -> Union[List[dict], Iterable[dict]]:
    """
    Args:
        file_path: jsonl file path
        return_iter: if the jsonl is really large, you can set `return_iter=True`
    """
    if not file_path.endswith('.jsonl'):
        raise FileTypeError(f'{file_path} is not a jsonl file!')

    def parse_fn():
        with open(file_path, 'r', encoding='utf-8') as json_file:
            for line in json_file:
                    yield json.loads(line)

    if return_iter:
        return parse_fn()
    else:
        return list(parse_fn())


def dump_jsonl(data: List[Any], file_path: str, json_encoder=None):
    """
    Args:
        file_path: dump jsonl file path
        data: data to dump
        json_encoder: when you want to save a custom class, you need to set `json_encoder`
    """
    if not file_path.endswith('.jsonl'):
        raise FileTypeError(f'{file_path} is not a jsonl file!')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        for item in data:
            line = json.dumps(item, ensure_ascii=False, default=json_encoder)
            json_file.write(line + '\n')


def read_txt(file_path: str, return_iter: bool = False) -> Union[List[str], Iterable[str]]:
    """
    Args:
        file_path: txt file path
        return_iter: if the txt file is very large, you can set return_iter=True
    """
    if not file_path.endswith('.txt'):
        raise FileTypeError(f'{file_path} is not txt file!')

    def parse_fn():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()

    if return_iter:
        return parse_fn()
    else:
        return list(parse_fn())


def dump_txt(data: List[str], file_path: str):
    """
    Args:
        data: need to save
        file_path: save file path
    """
    if not file_path.endswith('.txt'):
        raise FileTypeError(f'{file_path} is not txt file!')

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as txt_file:
        for line in data:
            txt_file.write(line.strip() + '\n')


def read_config(file_path: str, return_dict: bool = False) -> Union[DictConfig, dict]:
    """
    Args:
        file_path: yaml or yml config path
        return_dict: you can set `return_dict=True` to return dict, not DictConfig
    """
    if not file_path.endswith('.yaml') and not file_path.endswith('.yml'):
        raise FileTypeError(f'{file_path} is not a yaml / yml file!')

    with open(file_path, 'r', encoding='utf-8') as f:
        config = OmegaConf.load(f)
    if return_dict:
        config = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    return config


def dump_config(data: Union[dict, DictConfig], file_path: str):
    """
    Args:
        data: the config you want to save
        file_path: save path
    """
    if not file_path.endswith('.yaml') and not file_path.endswith('.yml'):
        raise FileTypeError(f'{file_path} is not a yaml / yml file!')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        OmegaConf.save(data, f)


def read_pkl(file_path: str):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def dump_pkl(data: Any, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def read_large_excel(file_path: str) -> Iterable[dict]:
    wb = load_workbook(file_path, read_only=True)
    for sheet in wb.sheetnames[:1]:
        ws = wb[sheet]
        _iter = ws.iter_rows()
        header = [cell.value for cell in next(_iter)]
        for row in _iter:
            data = dict(zip(header, (cell.value for cell in row)))
            yield data


def read_excel(file_path: str, return_iter: bool = False) -> Union[pd.DataFrame, Iterable[dict]]:
    """
    Args:
        file_path:
        return_iter: if the excel file very large, you can set `return_iter=True`
    """
    if not file_path.endswith('.xlsx') and not file_path.endswith('.xls'):
        raise FileTypeError(f'{file_path} is not a excel file')

    if return_iter:
        return read_large_excel(file_path)
    else:
        return pd.read_excel(file_path)


def dump_excel(data: pd.DataFrame, file_path: str, index: bool = True):
    """
    Args:
        data: need to save
        file_path: file save path
        index: used by `dataframe.to_excel()`
    """
    if not file_path.endswith('.xlsx') and not file_path.endswith('.xls'):
        raise FileTypeError(f'{file_path} is not a excel file')

    data.to_excel(file_path, index=index)


def read_csv(
    file_path: str,
    delimiter: str = ',',
    return_iter: bool = False
) -> Union[pd.DataFrame, Iterable[List[str]]]:
    """
    Args:
        file_path: csv or tsv file path
        delimiter: if tsv file, delimiter will be changed to \t
        return_iter: if the file is large, you can set `return_iter=True`
    """
    if not file_path.endswith('.csv') and not file_path.endswith('.tsv'):
        raise FileTypeError(f'{file_path} is not a csv / tsv file!')

    if file_path.endswith('.tsv'):
        delimiter = '\t'

    def parse_fn():
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                yield row

    if return_iter:
        return parse_fn()
    else:
        return pd.read_csv(file_path, delimiter=delimiter)


def dump_csv(data: pd.DataFrame, file_path: str, delimiter: str = ',', index: bool = True):
    """
    Args:
        data: need to save
        file_path: save path
        delimiter: if tsv file, delimiter will be changed to \t
        index: save index column or not
    """
    if file_path.endswith('.tsv'):
        delimiter = '\t'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(sep=delimiter, index=index)


def rm_file(file_path: str):
    """
    Args:
        file_path: the file path you want to delete
    """
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)


def rm_dir(dir_path: str, ignore_errors: bool = True, onerror=None):
    """
    Args:
        dir_path: the directory you want to delete
        ignore_errors: used by shutil.rmtree
        onerror: used by shutil.rmtree
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path, ignore_errors, onerror)


def clean_dir(dir_path: str, ignore_errors: bool = True, onerror=None):
    """
    Args:
        dir_path: the directory you want to clean
        ignore_errors: used by shutil.rmtree
        onerror: used by shutil.rmtree
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path, ignore_errors, onerror)
        os.makedirs(dir_path, exist_ok=True)


def get_file_ext(file_path: str, with_dot: bool = True):
    ext = os.path.splitext(os.path.basename(file_path))[-1]
    if with_dot:
        return ext
    return ext[1:]


class FileReader:
    _FILE_EXT_TO_FUNC = {
        'json': read_json,
        'jsonl': read_jsonl,
        'txt': read_txt,
        'yaml': read_config,
        'yml': read_config,
        'pkl': read_pkl,
        'xlsx': read_excel,
        'xls': read_excel,
        'csv': read_csv,
        'tsv': read_csv,
    }

    _EXT_PARAM_MAP = {
        'json': [],
        'jsonl': ['return_iter'],
        'txt': ['return_iter'],
        'yaml': ['return_dict'],
        'yml': ['return_dict'],
        'pkl': [],
        'xlsx': ['return_iter'],
        'xls': ['return_iter'],
        'csv': ['return_iter', 'delimiter'],
        'tsv': ['return_iter', 'delimiter'],
    }

    _return_iter = False
    _return_dict = False
    _delimiter = ','

    @classmethod
    def read(
        cls,
        file_path: str,
        return_iter: bool = None,
        return_dict: bool = None,
        delimiter: str = None
    ):
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise FileNotFoundError(f'{file_path} is not a valid file path')

        if return_iter is None:
            return_iter = cls._return_iter
        if return_dict is None:
            return_dict = cls._return_dict
        if delimiter is None:
            delimiter = cls._delimiter

        file_ext = get_file_ext(file_path, False)
        if file_ext not in cls._FILE_EXT_TO_FUNC:
            raise ValueError(f'Unsupported file extension: {file_ext}')

        kwargs = {'file_path': file_path}

        params_needed = cls._EXT_PARAM_MAP.get(file_ext, [])
        if 'return_iter' in params_needed:
            kwargs['return_iter'] = return_iter
        if 'return_dict' in params_needed:
            kwargs['return_dict'] = return_dict
        if 'delimiter' in params_needed:
            kwargs['delimiter'] = delimiter

        func = cls._FILE_EXT_TO_FUNC[file_ext]
        try:
            return func(**kwargs)
        except Exception as e:
            raise FileReadError(f'(Function {func.__name__}) Failed to read {file_path}: {e}')


class FileWriter:
    _FILE_EXT_TO_FUNC = {
        'json': dump_json,
        'jsonl': dump_jsonl,
        'txt': dump_txt,
        'yaml': dump_config,
        'yml': dump_config,
        'pkl': dump_pkl,
        'xlsx': dump_excel,
        'xls': dump_excel,
        'csv': dump_csv,
        'tsv': dump_csv
    }

    _EXT_PARAM_MAP = {
        'json': ['json_encoder'],
        'jsonl': ['json_encoder'],
        'txt': [],
        'yaml': [],
        'yml': [],
        'pkl': [],
        'xlsx': ['index'],
        'xls': ['index'],
        'csv': ['index', 'delimiter'],
        'tsv': ['index', 'delimiter'],
    }

    _json_encoder = None
    _index = False
    _delimiter = ','

    @classmethod
    def write(
        cls,
        data: Any,
        file_path: str,
        json_encoder: Callable = None,
        index: bool = None,
        delimiter: str = None
    ):
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("Invalid file path provided.")

        if index is None:
            index = cls._index
        if delimiter is None:
            delimiter = cls._delimiter

        file_ext = get_file_ext(file_path, False)
        if file_ext not in cls._FILE_EXT_TO_FUNC:
            raise NotImplementedError(f'Not implemented for file extension: {file_ext}')

        func = cls._FILE_EXT_TO_FUNC[file_ext]
        kwargs = {
            'file_path': file_path,
            'data': data
        }

        params_needed = cls._EXT_PARAM_MAP.get(file_ext, [])
        if 'json_encoder' in params_needed:
            kwargs['json_encoder'] = json_encoder
        if 'index' in params_needed:
            kwargs['index'] = index
        if 'delimiter' in params_needed:
            kwargs['delimiter'] = delimiter

        try:
            func(**kwargs)
        except Exception as e:
            raise FileWriteError(f'(Function {func.__name__}) Failed to write {file_path}: {e}')
