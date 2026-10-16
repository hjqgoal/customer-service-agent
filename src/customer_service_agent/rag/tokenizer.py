"""中文用 jieba 搜索模式分词，排序交给 SQLite FTS5 的 BM25。

文档和查询使用同一种分词方法；用户输入不会直接成为 FTS 查询语法。
"""

import re

import jieba

STOP_WORDS = {
    "的",
    "了",
    "吗",
    "呢",
    "啊",
    "我",
    "你",
    "我们",
    "你们",
    "这个",
    "那个",
    "怎么",
    "如何",
    "请问",
    "一下",
    "帮我",
    "是否",
    "什么",
    "可以",
    "应该",
    "想",
}


def tokenize(text: str) -> list[str]:
    return [
        word.lower()
        for word in jieba.cut_for_search(text, HMM=False)
        if word not in STOP_WORDS and re.fullmatch(r"[\w]+", word, flags=re.UNICODE)
    ]


def match_expression(query: str) -> str:
    # 每个词加双引号，OR只是程序固定的连接符，不执行用户提供的查询运算符。
    return " OR ".join(f'"{word}"' for word in dict.fromkeys(tokenize(query)))
