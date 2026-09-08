#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NLG 评估指标集合。

按指标族分四类：

- **overlap** —— 词面重叠（:func:`calc_bleu`、:func:`calc_meteor`、:func:`calc_rouge`、
  :func:`calc_greedy_match_score`），需要 hypothesis + references。
- **embedding** —— 基于词向量的语义相似度（:func:`calc_embedding_average_score`、
  :func:`calc_extrema_cosine_similar_score`、:func:`calc_greedy_match_score`），
  需要预训练词向量（GloVe 或自定义 ``word_2_emb``）。
- **pretrained** —— 基于预训练模型（:func:`calc_bert_score`、:func:`calc_bart_score`），
  需在 GPU 上加载模型。
- **diversity** —— 多样性（:func:`calc_distinct`），只看 hypothesis。

:meth:`NLGEvaluator.__call__` 是统一入口，按 :attr:`NLGEvaluator.metric_list` 串起来
调用，失败的指标会被跳过并在日志中记录。

依赖较重（transformers / torch / nltk），使用前需安装对应可选依赖
（``pip install "lwj_tools[nlgeval]"``）。

Example:
    >>> from lwj_tools.evaluators.nlg import NLGEvaluator, NLGMetric, BertScoreConfig
    >>> evaluator = NLGEvaluator(
    ...     metric_list=[NLGMetric.BLEU, NLGMetric.METEOR],
    ...     tokenizer=lambda s: s.split(),
    ... )
    >>> results = evaluator(
    ...     hypothesis=["a cat sat on the mat"],
    ...     references=["there is a cat on the mat"],
    ... )
"""

import math
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from bert_score import score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from ..common.logging import get_logger
from ..common.math import cosine_similarity
from ..common.files import load_glove
from ..common._typing import FilePath

LOGGER = get_logger('lwj_tools')


@unique
class NLGMetric(Enum):
    """支持的 NLG 评估指标枚举，``str(metric)`` 取值为 ``FUN_MAP`` 的查找键。"""

    BLEU = 'bleu'
    GLEU = 'gleu'
    METEOR = 'meteor'
    ROUGE = 'rouge'
    BERT_SCORE = 'bert_score'
    BART_SCORE = 'bart_score'
    DISTINCT = 'distinct'
    GREEDY_MATCH = 'greedy_match'
    COSINE_SIMILAR = 'cosine_similar'
    EXTREMA_COSINE_SIMILAR = 'extrema_cosine_similar'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


@dataclass
class BartScoreConfig:
    """:func:`calc_bart_score` 的运行参数。

    Attributes:
        name_or_path: HuggingFace 模型标识或本地路径。
        batch_size: 推理 batch size。
        device: 推理设备（如 ``"cuda:0"``、``"cpu"``）。
        max_length: tokenizer 最大长度（超过会被截断）。
    """

    name_or_path: str = 'facebook/bart-large-cnn'
    batch_size: int = 16
    device: str = 'cuda:0'
    max_length: int = 1024

    def to_dict(self):
        return asdict(self)


@dataclass
class BertScoreConfig:
    """:func:`calc_bert_score` 的运行参数（透传给 :func:`bert_score.score`）。

    Attributes:
        model_type: HuggingFace 模型标识或本地路径。本地路径需要显式给出
            ``num_layers``；HuggingFace 模型名可省略 ``num_layers``。
        num_layers: 取倒数第几层 hidden state。HF 模型名时由库自动决定。
        verbose: 是否打印进度信息。
        idf: 是否按 IDF 加权。
        device: 推理设备，``None`` 表示 ``cuda:0``。
        batch_size: 推理 batch size。
        nthreads: 后处理线程数。
        all_layers: 是否返回所有层的分数。
        lang: 分词语言（影响 tokenizer 选择）。
        return_hash: 是否同时返回哈希。
        rescale_with_baseline: 是否用基线分数做 rescale。
        baseline_path: rescale 基线文件路径。
        use_fast_tokenizer: 是否使用 fast tokenizer。
    """

    model_type: str
    num_layers: Optional[int] = None
    verbose: bool = False
    idf: bool = False
    device: Optional[str] = None  # default cuda:0
    batch_size: int = 64
    nthreads: int = 4
    all_layers: bool = False
    lang: Optional[str] = None
    return_hash: bool = False
    rescale_with_baseline: bool = False
    baseline_path: Optional[str] = None
    use_fast_tokenizer: bool = False

    def to_dict(self):
        return asdict(self)


class GLEU:
    """句级 GLEU（Google-BLEU）计算器。

    把每条 source 对应的多条 reference 预先聚合，按 ``order`` 阶 n-gram 做匹配与截断，
    在 :meth:`gleu_stats` 中按句产出 GLEU 公式所需的统计量。

    Attributes:
        order: n-gram 最大阶数。
        samples: source / reference 条数。
        refs_group: 按 source 索引分组的 references。
        ref_lens: 每条 reference 的 token 数。
    """

    def __init__(
        self,
        sources: List[str],
        references: List[List[str]],
        order: int = 4,
    ):
        self.order = order
        source_size = len(sources)
        ref_size = len(references[0])  # maybe you have more than one standard answer
        assert source_size == ref_size
        self.samples = source_size

        self.all_source_ngrams: List[List[Counter]] = []
        self.process_sources(sources)
        self.refs_group: List[List[str]] = []
        self.ref_lens: List[List[int]] = []
        self.all_ref_ngrams_freq: List[Counter] = [Counter() for _ in range(order)]
        self.all_ref_ngrams: List[List[Counter]] = [[] for _ in range(self.samples)]
        self.process_references(references)

    @staticmethod
    def get_n_gram(sentence: str, n) -> Counter:
        """把 ``sentence`` 切成 token 后返回 ``n`` 元组形式的 n-gram 频数。"""
        words = sentence.split()
        return Counter(
            [
                tuple(words[j:j + n])
                for j in range(len(words) + 1 - n)
            ],
        )

    @staticmethod
    def get_ngram_diff(a, b) -> Counter:
        """``a - b``（n-gram 集合差），返回仅出现在 ``a`` 中的频数。"""
        diff = Counter(a)
        for k in (set(a) & set(b)):
            del diff[k]
        return diff

    @staticmethod
    def gleu(stats, smooth=False):
        """根据 GLEU 统计量列表算最终 GLEU 分数。

        Args:
            stats: 长度 ``2 * order + 2`` 的统计量列表（参见 :meth:`gleu_stats`）。
            smooth: 为 ``True`` 时把 ``0`` 替换为 ``1`` 再算对数。
        """
        if smooth:
            stats = [s if s != 0 else 1 for s in stats]
        if len(list(filter(lambda x: x == 0, stats))) > 0:
            return 0
        c, r = stats[: 2]
        log_gleu_prec = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4
        return math.exp(min([0, 1 - float(r) / c]) + log_gleu_prec)

    def process_sources(self, sources: List[str]):
        """预处理 sources 的多阶 n-gram 频数。"""
        self.all_source_ngrams = [
            [GLEU.get_n_gram(s, n) for n in range(1, self.order + 1)]
            for s in sources
        ]

    def process_references(self, references: List[List[str]], ):
        """预处理 references：按 source 下标分组、统计每阶 n-gram 频数与 reference 长度。"""
        # references = [
        #     [r00, r01, r02, ...], reference file 0
        #     [r10, r11, r12, ...], reference file 1
        #     ...
        # ]
        # ref_groups: List[List[str]]= [
        #     [r00, r10, ...],
        #     [r01, r11, ...],
        #     [r02, r12, ...],
        #     ...
        # ]
        # ref_lens: List[List[int]] = [
        #     [len(r00.split()), len(r10.split()), ...],
        #     [len(r01.split()), len(r11.split()), ...],
        #     [len(r02.split()), len(r12.split()), ...],
        #     ...
        # ]

        for i in range(self.samples):
            self.refs_group.append([references[j][i] for j in range(len(references))])

        for i in range(self.samples):
            self.ref_lens.append([len(references[j][i].split()) for j in range(len(references))])

        for i, refs in enumerate(self.refs_group):
            for n in range(1, self.order + 1):
                ngrams: Counter = GLEU.get_n_gram(refs[0], n)
                self.all_ref_ngrams[i].append(ngrams)
                for k in ngrams.keys():
                    self.all_ref_ngrams_freq[n - 1][k] += 1
                for ref in refs[1:]:
                    new_ngrams = GLEU.get_n_gram(ref, n)
                    for nn in new_ngrams.elements():
                        if new_ngrams[nn] > ngrams.get(nn, 0):
                            ngrams[nn] = new_ngrams[nn]

    def normalization(self, ngram, n):
        """``ngram`` 在第 ``n`` 阶 reference 频数中的归一化值。"""
        return 1.0 * self.all_ref_ngrams_freq[n - 1][ngram] / len(self.ref_lens[0])

    def gleu_stats(self, hypothesis: str, hyp_ind: int, ref_ind: int):
        """对单条 hypothesis 生成 GLEU 统计量（hyp_len / ref_len / 各阶匹配数等）。"""
        hyp_len = len(hypothesis.split())
        hyp_ngrams = [GLEU.get_n_gram(hypothesis, n) for n in range(1, self.order + 1)]

        ref_len = self.ref_lens[hyp_ind][ref_ind]

        yield hyp_len
        yield ref_len

        for n in range(1, self.order + 1):
            h_ngrams = hyp_ngrams[n - 1]
            s_ngrams = self.all_source_ngrams[hyp_ind][n - 1]
            r_ngrams = GLEU.get_n_gram(self.refs_group[hyp_ind][ref_ind], n)
            s_ngram_diff = GLEU.get_ngram_diff(s_ngrams, r_ngrams)
            yield max(
                [
                    sum((h_ngrams & r_ngrams).values()) - sum((h_ngrams & s_ngram_diff).values()),
                    0
                ],
            )
            yield max([hyp_len + 1 - n, 0])


def calc_bleu(
    references: List[str],
    hypothesis: List[str],
    *,
    tokenizer: Callable = str.split,
    n: Union[int, List[int]] = 4,
    weights: Optional[List[Tuple[float, ...]]] = None,
    metrics: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """计算 BLEU（语料级和/或句级）。

    Args:
        references: 参考译文列表。
        hypothesis: 模型生成文本列表。
        tokenizer: token 切分函数，默认 ``str.split``。
        n: n-gram 阶数；可为 ``int`` 或多阶 ``List[int]``。
        weights: 各阶权重，与 ``n`` 一一对应。
        metrics: 输出哪些 BLEU 子项；``['corpus-bleu', 'sentence-bleu']`` 的子集。
        verbose: 是否打印进度条与日志。

    Returns:
        每个 metric 名到对应分数列表的映射。
    """

    if metrics is None:
        metrics = ['corpus-bleu', 'sentence-bleu']

    assert all(m in metrics for m in ['corpus-bleu', 'sentence-bleu']), \
        'The metrics should be "corpus-bleu" and "sentence-bleu"'

    n_samples = len(references)

    if verbose:
        LOGGER.info(f'Calculating BLEU: {metrics}')
        data_iter = tqdm(zip(hypothesis, references), total=n_samples, dynamic_ncols=True, desc='Tokenizing')
    else:
        data_iter = zip(hypothesis, references)

    hyp_texts, ref_texts = [], []
    for predict, reference in data_iter:
        hyp_texts.append(tokenizer(predict))
        ref_texts.append([tokenizer(reference)])

    ns = n
    if isinstance(ns, int):
        ns = [n]

    assert all(isinstance(n, int) and n > 0 for n in ns), \
        'The order should be an integer greater than 0'

    if weights is None:
        weights = [(1.0 / n,) * n for n in ns]

    if isinstance(weights, tuple):
        weights = [weights]

    assert len(weights) == len(ns), \
        'The number of weights and the number of orders are not consistent'

    results = {}

    if 'corpus-bleu' in metrics:
        LOGGER.info('Calculating corpus-bleu...')
        corpus_bleu_scores = corpus_bleu(
            list_of_references=ref_texts,
            hypotheses=hyp_texts,
            weights=weights,
        )
        results['corpus-bleu'] = corpus_bleu_scores

    if 'sentence-bleu' in metrics:
        LOGGER.info('Calculating sentence-bleu...')
        if verbose:
            data_iter = tqdm(zip(hyp_texts, ref_texts), total=n_samples, dynamic_ncols=True, desc='Sentence-BLEU')
        else:
            data_iter = zip(hyp_texts, ref_texts)
        sentence_bleu_scores = np.asarray([0.0] * len(weights))
        for hyp, ref in data_iter:
            cur_sent_scores = np.asarray(
                sentence_bleu(
                    references=ref,
                    hypothesis=hyp,
                    weights=weights,
                ),
            )
            sentence_bleu_scores = sentence_bleu_scores + cur_sent_scores
        sentence_bleu_scores /= n_samples
        sentence_bleu_scores = sentence_bleu_scores.tolist()
        results['sentence-bleu'] = sentence_bleu_scores

    return results


def calc_rouge(
    references: List[str],
    hypothesis: List[str],
    tokenizer: Callable[[str], List[str]] = str.split,
    metrics: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """计算 ROUGE 分数（基于 :class:`rouge.Rouge`）。

    Args:
        references: 参考文本列表。
        hypothesis: 生成文本列表。
        tokenizer: token 切分函数。
        metrics: 需要的子项，如 ``['rouge-1', 'rouge-2', 'rouge-l']``。
        verbose: 是否打印进度条与日志。

    Returns:
        ROUGE 子项到分数的映射（f/p/r 各一组，取均值）。
    """

    if metrics is None:
        metrics = ['rouge-1', 'rouge-2', 'rouge-l']

    if verbose:
        LOGGER.info(f'Calculating ROUGE: {metrics}')
        data_iter = tqdm(zip(hypothesis, references), total=len(hypothesis), dynamic_ncols=True, desc='Tokenizing')
    else:
        data_iter = zip(hypothesis, references)

    ref_texts, hyp_texts = [], []
    for hyp, ref in data_iter:
        hyp_texts.append(' '.join(tokenizer(hyp)))
        ref_texts.append(' '.join(tokenizer(ref)))

    rouge_score = Rouge(metrics).get_scores(hyp_texts, ref_texts, avg=True)
    return rouge_score


def calc_meteor(
    references: List[str],
    hypothesis: List[str],
    tokenizer: Callable = str.split,
    verbose: bool = False,
) -> float:
    """计算 METEOR 分数（基于 :func:`nltk.translate.meteor_score.meteor_score`）。

    Args:
        references: 参考文本列表。
        hypothesis: 生成文本列表。
        tokenizer: token 切分函数。
        verbose: 是否打印进度条。

    Returns:
        所有样本 METEOR 的平均值。
    """
    n_samples = len(references)

    if verbose:
        LOGGER.info('Calculating METEOR...')
        data_iter = tqdm(
            zip(hypothesis, references), total=n_samples, dynamic_ncols=True, desc='Calculating METEOR ...',
        )
    else:
        data_iter = zip(hypothesis, references)

    total_meteor_score = 0
    for hyp, ref in data_iter:
        total_meteor_score += meteor_score([tokenizer(ref)], tokenizer(hyp))
    meteor = total_meteor_score / n_samples
    return meteor


def calc_gleu(
    sources: List[str],
    references: List[str],
    hypothesis: List[str],
    n: Union[int, List[int]] = 4,
    verbose: bool = False,
) -> List[float]:
    """计算 Google-BLEU（GLEU）。

    Args:
        sources: 源文本（如翻译任务的原文）。
        references: 参考文本。
        hypothesis: 生成文本。
        n: n-gram 阶数。
        verbose: 是否打印进度条。

    Returns:
        每个 ``n`` 对应一个 GLEU 分数。
    """
    n_samples = len(sources)
    ns = [n] if isinstance(n, int) else n
    assert all(n > 0 and isinstance(n, int) for n in ns), \
        'The order should be an integer greater than 0'

    if verbose:
        LOGGER.info(f'Calculating GLEU: {n}')

    gleu_scores = []
    for n_order in ns:
        gleu_calculator = GLEU(sources, [references], n_order)
        indices = [0] * n_samples
        stats = [0] * len(range(2 * n_order + 2))

        if verbose:
            data_iter = tqdm(enumerate(hypothesis), total=n_samples, dynamic_ncols=True, desc=f'GLEU-{n_order}')
        else:
            data_iter = enumerate(hypothesis)

        for i, hyp in data_iter:
            stats = [sum(scores) for scores in zip(
                stats,
                [s for s in gleu_calculator.gleu_stats(hyp, i, indices[i])],
            )]

        gleu_scores.append(GLEU.gleu(stats))
    return gleu_scores


def calc_bert_score(
    references: List[str],
    hypothesis: List[str],
    score_config: BertScoreConfig,
    reduction: str = 'mean',
    round_bits: int = 6,
    iter_size: int = 5000,
    verbose: bool = False,
) -> Union[Dict[str, float], Dict[str, List[float]]]:
    """计算 BERTScore。

    Args:
        references: 参考文本列表。
        hypothesis: 生成文本列表。
        score_config: :class:`BertScoreConfig`。
        reduction: ``'mean'`` / ``'sum'`` / ``'none'``。
        round_bits: 保留小数位数（仅 ``reduction != 'none'`` 时生效）。
        iter_size: 每批处理的样本数，防止显存溢出。
        verbose: 是否打印进度条。

    Returns:
        ``{'P', 'R', 'F'}`` 到对应分数的映射。
    """
    n_samples = len(references)
    P, R, F = [], [], []

    if verbose:
        LOGGER.info('Calculating BERTScore...')
        data_iter = tqdm(range(0, n_samples, iter_size), dynamic_ncols=True, desc='Calculating BERTScore ...')
    else:
        data_iter = range(0, n_samples, iter_size)
    for i in data_iter:
        p, r, f = score(
            cands=hypothesis[i:i + iter_size],
            refs=references[i:i + iter_size],
            **score_config.to_dict(),
        )
        P.extend(p.tolist())
        R.extend(r.tolist())
        F.extend(f.tolist())

    if reduction is None:
        reduction = 'none'

    reduction = reduction.lower()
    if reduction != 'none':
        if reduction == 'mean':
            P = sum(P) / n_samples
            R = sum(R) / n_samples
            F = sum(F) / n_samples
        elif reduction == 'sum':
            P = sum(P)
            R = sum(R)
            F = sum(F)
        else:
            raise ValueError(f'Unknown reduction = {reduction}')

        P = round(P, round_bits)
        R = round(R, round_bits)
        F = round(F, round_bits)

    return {
        'P': P,
        'R': R,
        'F': F,
    }


@torch.no_grad()
def calc_bart_score(
    references: List[str],
    hypothesis: List[str],
    score_config: BartScoreConfig,
    verbose: bool = False,
) -> float:
    """计算 BARTScore（基于 :class:`~transformers.BartForConditionalGeneration`）。

    Args:
        references: 参考文本列表。
        hypothesis: 生成文本列表。
        score_config: :class:`BartScoreConfig`。
        verbose: 是否打印进度条。

    Returns:
        所有样本 BARTScore 的平均值。失败样本跳过并记录日志。
    """

    if verbose:
        LOGGER.info('Calculating BARTScore...')

    device = score_config.device
    batch_size = score_config.batch_size

    if verbose:
        LOGGER.info(f'Loading model from {score_config.name_or_path} ...')
    model = BartForConditionalGeneration.from_pretrained(score_config.name_or_path).to(device).eval()
    tokenizer = BartTokenizer.from_pretrained(score_config.name_or_path)
    loss_fct = nn.NLLLoss(ignore_index=model.config.pad_token_id, reduction='none')
    lsm = nn.LogSoftmax(dim=1)

    n_samples = len(references)

    if verbose:
        data_iter = tqdm(range(0, n_samples, batch_size), dynamic_ncols=True, desc='Calculating BARTScore ...')
    else:
        data_iter = range(0, n_samples, batch_size)

    score_list = []
    for i in data_iter:
        src_list = hypothesis[i: i + batch_size]
        tgt_list = references[i: i + batch_size]

        try:
            encoded_src = tokenizer(
                src_list,
                max_length=score_config.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt',
            )
            encoded_tgt = tokenizer(
                tgt_list,
                max_length=score_config.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt',
            )

            src_tokens = encoded_src['input_ids'].to(device)
            src_mask = encoded_src['attention_mask'].to(device)

            tgt_tokens = encoded_tgt['input_ids'].to(device)
            tgt_mask = encoded_tgt['attention_mask']
            tgt_len = tgt_mask.sum(dim=1).to(device)

            output = model(
                input_ids=src_tokens,
                attention_mask=src_mask,
                labels=tgt_tokens,
            )

            logits = output.logits.view(-1, model.config.vocab_size)
            loss = loss_fct(lsm(logits), tgt_tokens.view(-1))
            loss = loss.view(tgt_tokens.shape[0], -1)  # [bsz, tgt_len]
            loss = loss.sum(dim=1) / tgt_len
            curr_score_list = [-x.item() for x in loss]
            score_list += curr_score_list

        except Exception as e:
            traceback.print_exc()
            LOGGER.error(e)

    bart_score = sum(score_list) / n_samples
    return bart_score


def calc_greedy_match_score(
    references: List[str],
    hypothesis: List[str],
    word_2_emb: Dict[str, np.ndarray],
    tokenizer: Optional[Callable] = str.split,
    unk_emb: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> float:
    """Greedy Matching：对每个 hyp token 取与 reference 的最大余弦相似度并平均。

    Args:
        references: 参考文本列表。
        hypothesis: 生成文本列表。
        word_2_emb: 词到 embedding 的映射。
        tokenizer: token 切分函数。
        unk_emb: 未登录词回退向量；``None`` 时用全零向量。
        verbose: 是否打印进度条。

    Returns:
        所有样本 greedy match 分数的均值。
    """
    emb_dim = list(word_2_emb.values())[0].shape[0]
    if unk_emb is None:
        unk_emb = np.zeros(emb_dim, dtype=np.float32)

    if verbose:
        LOGGER.info('Calculating Greedy Match Score...')
        data_iter = tqdm(
            zip(hypothesis, references),
            total=len(references),
            dynamic_ncols=True,
            desc='Calculating Greedy Match Score ...',
        )
    else:
        data_iter = zip(hypothesis, references)

    scores = []
    for hyp, ref in data_iter:
        # (seq_len, emb)
        hyp_emb = np.vstack([word_2_emb.get(hyp_token, unk_emb) for hyp_token in tokenizer(hyp)])
        ref_emb = np.vstack([word_2_emb.get(ref_token, unk_emb) for ref_token in tokenizer(ref)])

        # sim_matrix.shape = (hyp_len, ref_len)
        sim_matrix = cosine_similarity(hyp_emb, ref_emb)
        scores.append(sim_matrix.max())

    return np.mean(scores).item()


def calc_embedding_average_score(
    references: List[str],
    hypothesis: List[str],
    word_2_emb: Dict[str, np.ndarray],
    tokenizer: Optional[Callable] = str.split,
    unk_emb: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> float:
    """Embedding Average：句向量为词向量均值，与 reference 做余弦相似度。

    Args:
        references: 参考文本列表。
        hypothesis: 生成文本列表。
        word_2_emb: 词到 embedding 的映射。
        tokenizer: token 切分函数。
        unk_emb: 未登录词回退向量；``None`` 时用全零向量。
        verbose: 是否打印进度条。

    Returns:
        所有样本 embedding-average 分数的均值。
    """
    emb_dim = list(word_2_emb.values())[0].shape[0]
    if unk_emb is None:
        unk_emb = np.zeros(emb_dim, dtype=np.float32)

    if verbose:
        LOGGER.info('Calculating Embedding Average Score...')
        data_iter = tqdm(
            zip(hypothesis, references),
            total=len(references),
            dynamic_ncols=True,
            desc='Calculating Embedding Average Score...',
        )
    else:
        data_iter = zip(hypothesis, references)

    scores = []
    for hyp, ref in data_iter:
        # (seq_len, emb) -> (1, emb)
        hyp_emb = np.vstack([word_2_emb.get(hyp_token, unk_emb) for hyp_token in tokenizer(hyp)])
        hyp_emb = hyp_emb.mean(axis=0, keepdims=True)

        ref_emb = np.vstack([word_2_emb.get(ref_token, unk_emb) for ref_token in tokenizer(ref)])
        ref_emb = ref_emb.mean(axis=0, keepdims=True)

        # sim_matrix.shape = (1, 1)
        sim_matrix = cosine_similarity(hyp_emb, ref_emb)
        scores.append(sim_matrix[0][0])

    return np.mean(scores).item()


def calc_extrema_cosine_similar_score(
    references: List[str],
    hypothesis: List[str],
    word_2_emb: Dict[str, np.ndarray],
    tokenizer: Optional[Callable] = str.split,
    unk_emb: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> float:
    """Extrema：句向量为每个维度上的极值（max），与 reference 做余弦相似度。

    Args:
        references: 参考文本列表。
        hypothesis: 生成文本列表。
        word_2_emb: 词到 embedding 的映射。
        tokenizer: token 切分函数。
        unk_emb: 未登录词回退向量；``None`` 时用全零向量。
        verbose: 是否打印进度条。

    Returns:
        所有样本 extrema 分数的均值。
    """
    emb_dim = list(word_2_emb.values())[0].shape[0]
    if unk_emb is None:
        unk_emb = np.zeros(emb_dim, dtype=np.float32)

    if verbose:
        LOGGER.info('Calculating Extrema Cosine Similar Score...')
        data_iter = tqdm(
            zip(hypothesis, references),
            total=len(references),
            dynamic_ncols=True,
            desc='Calculating Extrema Cosine Similar Score...',
        )
    else:
        data_iter = zip(hypothesis, references)

    scores = []
    for hyp, ref in data_iter:
        # (seq_len, emb) -> (1, emb)
        hyp_emb = np.vstack([word_2_emb.get(hyp_token, unk_emb) for hyp_token in tokenizer(hyp)])
        hyp_emb = hyp_emb.max(axis=0, keepdims=True)

        ref_emb = np.vstack([word_2_emb.get(ref_token, unk_emb) for ref_token in tokenizer(ref)])
        ref_emb = ref_emb.max(axis=0, keepdims=True)

        # sim_matrix.shape = (1, 1)
        sim_matrix = cosine_similarity(hyp_emb, ref_emb)
        scores.append(sim_matrix[0][0])

    return np.mean(scores).item()


def calc_distinct(
    hypothesis: List[str],
    tokenizer: Callable = str.split,
    n: Union[int, List[int]] = 4,
    verbose: bool = False,
) -> List[float]:
    """计算 Distinct-n：unique n-gram 数 / 总 n-gram 数。

    Args:
        hypothesis: 生成文本列表。
        tokenizer: token 切分函数。
        n: n-gram 阶数。
        verbose: 是否打印进度条。

    Returns:
        每个 ``n`` 对应一个 distinct 分数。
    """
    ns = [n] if isinstance(n, int) else n
    assert all(n > 0 and isinstance(n, int) for n in ns), 'The order should be an integer greater than 0.'

    ngrams: Dict[int, Counter] = {n: Counter() for n in ns}

    if verbose:
        LOGGER.info(f'Calculating Distinct: {ns}')
        data_iter = tqdm(
            hypothesis,
            total=len(hypothesis),
            dynamic_ncols=True,
            desc='Calculating Distinct ...',
        )
    else:
        data_iter = hypothesis

    for hyp in data_iter:
        hyp_words: List[str] = tokenizer(hyp)
        for n in ns:
            for i in range(len(hyp_words) - n + 1):
                ngrams[n][tuple(hyp_words[i:i + n])] += 1

    distinct_scores = [len(ngrams[n]) / sum(ngrams[n].values()) for n in ns]
    return distinct_scores


class NLGEvaluator:
    """批量 NLG 指标计算器：按 :attr:`metric_list` 调用对应 :func:`calc_*`。

    Attributes:
        metric_list: 实际要计算的指标集合（已剔除 :attr:`omit_metric_list`）。
        tokenizer: 过滤掉空串后的 token 切分函数。
        bert_score_config: BERTScore 使用的 :class:`BertScoreConfig`。
        bart_score_config: BARTScore 使用的 :class:`BartScoreConfig`。
        word_2_emb: 加载后的词向量映射（仅 embedding 类指标需要）。
        verbose: 是否打印进度条。
    """

    METRICS = {
        'overlap': [NLGMetric.BLEU, NLGMetric.METEOR, NLGMetric.ROUGE, NLGMetric.GLEU],
        'embedding': [NLGMetric.GREEDY_MATCH, NLGMetric.COSINE_SIMILAR, NLGMetric.EXTREMA_COSINE_SIMILAR],
        'pretrained': [NLGMetric.BART_SCORE, NLGMetric.BERT_SCORE],
        'diversity': [NLGMetric.DISTINCT],
    }

    FUN_MAP = {
        NLGMetric.BLEU: calc_bleu,
        NLGMetric.ROUGE: calc_rouge,
        NLGMetric.GLEU: calc_gleu,
        NLGMetric.METEOR: calc_meteor,
        NLGMetric.BERT_SCORE: calc_bert_score,
        NLGMetric.BART_SCORE: calc_bart_score,
        NLGMetric.GREEDY_MATCH: calc_greedy_match_score,
        NLGMetric.COSINE_SIMILAR: calc_extrema_cosine_similar_score,
        NLGMetric.EXTREMA_COSINE_SIMILAR: calc_extrema_cosine_similar_score,
        NLGMetric.DISTINCT: calc_distinct,
    }

    def __init__(
        self,
        metric_list: List[NLGMetric],
        tokenizer: Callable[[str], List[str]] = str.split,
        use_overlap: Optional[bool] = None,
        use_embedding: Optional[bool] = None,
        use_pretrained: Optional[bool] = None,
        use_diversity: Optional[bool] = None,
        omit_metric_list: Optional[List[NLGMetric]] = None,
        bert_score_config: Optional[BertScoreConfig] = None,
        bart_score_config: Optional[BartScoreConfig] = None,
        glove_path: Optional[FilePath] = None,
        word_2_emb: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        glove_skip_first_row: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            metric_list: 实际要计算的指标。
            tokenizer: token 切分函数，默认 ``str.split``。
            use_overlap: ``False`` 时跳过 overlap 类指标。
            use_embedding: ``False`` 时跳过 embedding 类指标。
            use_pretrained: ``False`` 时跳过 pretrained 类指标。
            use_diversity: ``False`` 时跳过 diversity 类指标。
            omit_metric_list: 显式剔除的指标集合（优先级高于上面开关）。
            bert_score_config: BERTScore 配置（不传则用默认 ``roberta-large``）。
            bart_score_config: BARTScore 配置（不传则用默认 ``facebook/bart-large-cnn``）。
            glove_path: GloVe 文件路径（与 ``word_2_emb`` 二选一）。
            word_2_emb: 自定义词向量字典（优先级高于 ``glove_path``）。
            glove_skip_first_row: GloVe 首行是否为 header。
            verbose: 是否打印进度条与日志。
        """
        if omit_metric_list is None:
            omit_metric_list = []

        if use_overlap is not None and not use_overlap:
            omit_metric_list.extend(self.METRICS['overlap'])

        if use_embedding is not None and not use_embedding:
            omit_metric_list.extend(self.METRICS['embedding'])

        if use_pretrained is not None and not use_pretrained:
            omit_metric_list.extend(self.METRICS['pretrained'])

        if use_diversity is not None and not use_diversity:
            omit_metric_list.extend(self.METRICS['diversity'])

        self.metric_list = [metric for metric in metric_list if metric not in omit_metric_list]

        assert len(self.metric_list) > 0, 'No indicators were specified'

        self.word_2_emb = None
        if use_embedding:
            assert glove_path or word_2_emb, \
                'Use the word vector metric, glove_path and word_2_emb provide at least one'

            if word_2_emb is not None:
                self.word_2_emb = word_2_emb
            else:
                LOGGER.info(f'Loading glove file from : {glove_path}')
                self.word_2_emb = load_glove(glove_path, glove_skip_first_row)

        self.tokenizer = lambda s: list(filter(lambda token: len(token.strip()) > 0, tokenizer(s)))

        self.bert_score_config = bert_score_config if bert_score_config else BertScoreConfig('roberta-large')
        self.bart_score_config = bart_score_config if bart_score_config else BartScoreConfig()

        self.verbose = verbose

    def __call__(
        self,
        hypothesis: List[str],
        references: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        specific_config: Optional[Dict[NLGMetric, Dict[str, Any]]] = None
    ):
        """批量计算多种自然语言生成（NLG）评估指标。

        该方法用于对生成文本（`hypothesis`）进行多维度自动评估，支持与参考文本（`references`）
        和源文本（`sources`，如翻译任务中的原文）对比，并可通过 `specific_config` 为每种指标
        提供自定义配置。

        Args:
            hypothesis (List[str]): 待评估的生成文本列表。
            references (Optional[List[str]]): 参考文本（标准答案）列表。大多数指标（如 BLEU、ROUGE）
                必须提供，且长度需与 `hypothesis` 一致。
            sources (Optional[List[str]]): 源输入文本列表（例如机器翻译中的原文）。仅 GLEU 指标需要，
                若提供则长度必须与 `hypothesis` 一致。
            specific_config (Optional[Dict[NLGMetric, Dict[str, Any]]]): 各指标的自定义配置字典。
                键为指标枚举值（如 `NLGMetric.BLEU`），值为传递给具体指标函数的参数字典。

        Returns:
            Dict[str, Any]: 以指标名称（字符串）为键、计算结果为值的字典。若某指标计算失败，
            该指标将被跳过，并在日志中记录错误信息。

        Examples:
            >>> # 配置 BLEU 计算 1~4 元语法
            >>> config = {}
            >>> config[NLGMetric.BLEU] = {'n': 4}
            >>> # 或显式指定各阶权重
            >>> config[NLGMetric.BLEU] = {
            ...     'weights': [
            ...         (1.0,),
            ...         (0.5, 0.5),
            ...         (1/3, 1/3, 1/3),
            ...         (0.25, 0.25, 0.25, 0.25)
            ...     ]
            ... }
            >>> # 同时计算语料级和句子级 BLEU
            >>> config[NLGMetric.BLEU] = {'metrics': ['corpus-bleu', 'sentence-bleu']}

            >>> # 配置 ROUGE 指标（仅支持 rouge-1 至 rouge-5 及 rouge-l）
            >>> config[NLGMetric.ROUGE] = {'metrics': ['rouge-1', 'rouge-2', 'rouge-l']}

            >>> # 配置 GLEU 或 Distinct 的 n-gram 阶数
            >>> config[NLGMetric.GLEU] = {'n': [1, 2, 3, 4]}
            >>> config[NLGMetric.DISTINCT] = {'n': 2}

            >>> # 配置 BERTScore：启用批处理防止内存溢出
            >>> config[NLGMetric.BERT_SCORE] = {
            ...     'reduction': 'mean',      # 结果聚合方式
            ...     'iter_size': 5000,        # 每批处理 5000 个样本
            ...     'round_bits': 6           # 保留小数位数
            ... }

            >>> # 使用示例
            >>> evaluator = YourEvaluatorClass()
            >>> results = evaluator(
            ...     hypothesis=['你好 世界'],
            ...     references=['你好 世界'],
            ...     specific_config=config
            ... )
        """

        assert len(hypothesis) == len(references), \
            'hypothesis and references must have the same length'

        result = {}
        for metric in self.metric_list:
            try:
                func = self.FUN_MAP[metric]
                func_kwargs: Dict[str, Any] = {
                    'hypothesis': hypothesis,
                    'references': references,
                    'verbose': self.verbose
                }

                if metric == NLGMetric.DISTINCT:
                    func_kwargs.pop('references')

                if metric in self.METRICS['overlap']:
                    func_kwargs['tokenizer'] = self.tokenizer

                    if metric == NLGMetric.GLEU:
                        assert sources is not None, ('GLEU requires sources')
                        assert len(sources) == len(hypothesis)
                        func_kwargs['sources'] = sources

                if metric in self.METRICS['pretrained']:
                    func_kwargs['score_config'] = self.bert_score_config \
                        if metric == NLGMetric.BERT_SCORE \
                        else self.bart_score_config

                if metric in self.METRICS['embedding']:
                    func_kwargs['word_2_emb'] = self.word_2_emb
                    func_kwargs['tokenizer'] = self.tokenizer

                if specific_config.get(metric, None):
                    func_kwargs.update(specific_config[metric])

                result[str(metric)] = func(**func_kwargs)
            except Exception as e:
                LOGGER.error(f'The calculation of the evaluation indicators failed, metric:{metric}')
                LOGGER.error(e)

        return result
