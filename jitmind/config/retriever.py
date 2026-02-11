from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Union, List, Optional


@dataclass
class DenseRetrieverConfig:
    """Dense vector retriever configuration."""
    model_name: str = "BAAI/bge-m3"
    model_class: str = "encoder-only-base"
    normalize_embeddings: bool = True
    pooling_method: str = "cls"
    trust_remote_code: bool = True
    query_instruction_for_retrieval: Optional[str] = None
    use_fp16: bool = False
    devices: List[str] = field(default_factory=lambda: ["cuda:0"])
    batch_size: int = 32
    max_length: int = 512
    index_dir: str = "./index/dense"
    api_url: Optional[str] = None


@dataclass
class IndexRetrieverConfig:
    """Index retriever configuration."""
    index_dir: str = "./index/index"


@dataclass
class BM25RetrieverConfig:
    """BM25 keyword retriever configuration."""
    index_dir: str = "./index/bm25"
    threads: int = 4


@dataclass
class CohereEmbedRetrieverConfig:
    """Cohere embedding retriever configuration"""
    api_key: Optional[str] = None
    base_url: str = "https://api.cohere.com"
    model_name: str = "embed-v4.0"
    input_type_doc: str = "search_document"
    input_type_query: str = "search_query"
    batch_size: int = 32
    max_length: int = 512
    index_dir: str = "./index/cohere_dense"


@dataclass
class CohereRerankerConfig:
    """Cohere reranker configuration"""
    api_key: Optional[str] = None
    base_url: str = "https://api.cohere.com"
    model_name: str = "rerank-v4.0-pro"
    top_k: int = 20
