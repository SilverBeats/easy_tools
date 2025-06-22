import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any, Callable, Optional
import argparse
from tqdm import tqdm
import logging
import regex
import hashlib
import uuid

__all__ = [
	'random_choice',
	'str2bool',
	'get_logger',
	'MultiWorkerRunner',
	'get_dir_file_path',
	'camel_to_snake',
	'shuffle',
	'get_uuid',
	'get_md5_id'
]


def random_choice(arr: List[Any], n: int = 1) -> List[Any]:
	"""
	random choice n elements from array
	Args:
		arr: array of elements
		n: the number of elements you want to select
	"""
	return random.sample(arr, min(n, len(arr)))


def str2bool(v):
	"""Used in argparse, when you want to set a bool parameter """
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')


class MultiWorkerRunner:
	"""
	You can use this class to run multiple workers
	"""

	def __init__(self, num_workers: int = -1, use_pbar: bool = True):
		if num_workers == -1:
			self.num_workers = os.cpu_count()
		else:
			self.num_workers = num_workers

		self.use_pbar = use_pbar

	def _single_worker(self, samples, worker_func, collection_func, pbar):
		for sample in samples:
			result = worker_func(sample)
			if collection_func is not None:
				collection_func(result)
			if pbar is not None:
				pbar.update(1)
				pbar.refresh()

	def _multi_workers(self, samples, worker_func, collection_func, pbar):
		with ThreadPoolExecutor(self.num_workers) as executor:
			tasks = [executor.submit(worker_func, sample) for sample in samples]
			for task in as_completed(tasks):
				result = task.result()
				if collection_func is not None:
					collection_func(result)
				if pbar is not None:
					pbar.update(1)
					pbar.refresh()

	def __call__(
		self,
		samples: List[Any],
		worker_func: Callable,
		collection_func: Callable = None,
		desc: str = "Running",
	):
		pbar = None
		if self.use_pbar:
			pbar = tqdm(total=len(samples), dynamic_ncols=True, desc=desc)
		if self.num_workers == 1:
			self._single_worker(samples, worker_func, collection_func, pbar)
		else:
			self._multi_workers(samples, worker_func, collection_func, pbar)
		if pbar is not None:
			pbar.close()


def get_logger(
	name: str,
	level: str = "info",
	formatter: Optional[str] = None,
	log_path: Optional[str] = None,
) -> logging.Logger:
	"""get a logger

	Args:
		name: logger name
		level: log level. Defaults to "info".
		formatter: log formatter. Defaults to None.
		log_path: log file path. Defaults to None.
	"""
	LEVELS = {
		"debug": logging.DEBUG,
		"info": logging.INFO,
		"warn": logging.WARN,
		"error": logging.ERROR,
		"fatal": logging.FATAL,
	}

	assert level in LEVELS

	logger = logging.getLogger(name)

	if not logger.handlers:
		level_ = LEVELS[level]
		logger.setLevel(level_)

		fmt = (
			formatter
			if formatter is not None
			else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
		)
		log_formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

		ch = logging.StreamHandler()
		ch.setLevel(level_)
		ch.setFormatter(log_formatter)
		logger.addHandler(ch)

		if log_path is not None:
			dirname = os.path.dirname(log_path)
			os.makedirs(dirname, exist_ok=True)
			fh = logging.FileHandler(log_path, encoding="utf-8")
			fh.setLevel(level_)
			fh.setFormatter(log_formatter)
			logger.addHandler(fh)

	return logger


def get_dir_file_path(
	dir_name: str,
	file_exts: Optional[List[str]] = None,
	skip_dir_names: Optional[List[str]] = None,
	skip_file_names: Optional[List[str]] = None,
	is_abs: Optional[bool] = False,
) -> List[str]:
	"""
	A function to scan a specify file dictionary and return a list of file paths
	Args:
		dir_name: dictionary name
		file_exts: which type file you want to get
		skip_dir_names: maybe you want to skip some sub-dictionary
		skip_file_names: maybe you want to skip some file
		is_abs: return absolute path or relative path
	"""
	if not os.path.isdir(dir_name):
		return []

	dir_name = os.path.realpath(os.path.abspath(dir_name) if is_abs else dir_name)

	file_exts = set(file_exts or [])
	skip_dir_names = set(skip_dir_names or [])
	skip_file_names = set(skip_file_names or [])

	arr = []
	stack = [dir_name]

	while stack:
		current_dir = stack.pop()
		try:
			with os.scandir(current_dir) as it:
				for entry in it:
					name = entry.name
					full_path = entry.path

					if entry.is_dir():
						if name in skip_dir_names:
							continue
						child_dir = os.path.realpath(full_path)
						if not child_dir.startswith(dir_name + os.sep):
							continue
						stack.append(child_dir)
					else:
						if name in skip_file_names:
							continue
						ext = os.path.splitext(name)[1]
						if file_exts and ext not in file_exts:
							continue
						arr.append(full_path)
		except (PermissionError, OSError) as e:
			continue

	return arr


def camel_to_snake(name: str) -> str:
	"""use this function to change a camel style name to snake style name"""
	s1 = regex.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
	return regex.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def shuffle(arr: List[Any], n):
	"""shuffle a list"""
	for _ in range(n):
		random.shuffle(arr)


def get_uuid(prefix: Optional[str] = None) -> str:
	"""return uuid"""
	if prefix is not None:
		return f"{prefix}-{uuid.uuid4().hex}"
	return uuid.uuid4().hex


def get_md5_id(text: str) -> str:
	"""return text's md5 value"""
	hash_str = hashlib.md5(text.encode("utf-8")).hexdigest()
	return hash_str
