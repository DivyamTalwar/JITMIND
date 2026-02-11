from __future__ import annotations

from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import os
import re
import html

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

from .documents import Document


class BaseLoader:
    def load(self) -> List[Document]:
        raise NotImplementedError


class TextFileLoader(BaseLoader):
    def __init__(self, path: str, source: Optional[str] = None, encoding: str = "utf-8") -> None:
        self.path = path
        self.source = source or path
        self.encoding = encoding

    def load(self) -> List[Document]:
        p = Path(self.path)
        if not p.exists() or not p.is_file():
            return []
        content = p.read_text(encoding=self.encoding, errors="ignore")
        return [Document(content=content, source=self.source, doc_id=p.stem, metadata={"path": str(p)})]


class DirectoryLoader(BaseLoader):
    def __init__(self, directory: str, glob: str = "**/*", include_ext: Optional[List[str]] = None) -> None:
        self.directory = directory
        self.glob = glob
        self.include_ext = [e.lower() for e in include_ext] if include_ext else None

    def load(self) -> List[Document]:
        root = Path(self.directory)
        if not root.exists():
            return []
        docs: List[Document] = []
        for p in root.glob(self.glob):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if self.include_ext and ext not in self.include_ext:
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            docs.append(Document(content=content, source=str(p), doc_id=p.stem, metadata={"path": str(p)}))
        return docs


class URLLoader(BaseLoader):
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 20) -> None:
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout

    def load(self) -> List[Document]:
        if requests is None:
            raise ImportError("requests is required for URLLoader. Install with: pip install requests")
        resp = requests.get(self.url, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        text = resp.text
        if "html" in content_type.lower():
            text = _strip_html(text)
        return [Document(content=text, source=self.url, doc_id=self._doc_id(), metadata={"url": self.url})]

    def _doc_id(self) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "-", self.url).strip("-")[:64]


class JSONLLoader(BaseLoader):
    def __init__(self, path: str, text_key: str = "text", id_key: str = "id") -> None:
        self.path = path
        self.text_key = text_key
        self.id_key = id_key

    def load(self) -> List[Document]:
        p = Path(self.path)
        if not p.exists() or not p.is_file():
            return []
        docs: List[Document] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                text = str(row.get(self.text_key, "")).strip()
                if not text:
                    continue
                doc_id = row.get(self.id_key)
                docs.append(Document(content=text, source=str(p), doc_id=str(doc_id) if doc_id else None, metadata=row))
        return docs


class S3Loader(BaseLoader):
    def __init__(self, bucket: str, prefix: str = "", region: Optional[str] = None) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.region = region

    def load(self) -> List[Document]:
        try:
            import boto3  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("boto3 is required for S3Loader. Install with: pip install boto3") from e
        s3 = boto3.client("s3", region_name=self.region)
        paginator = s3.get_paginator("list_objects_v2")
        docs: List[Document] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []) or []:
                key = obj.get("Key")
                if not key or key.endswith("/"):
                    continue
                body = s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                try:
                    content = body.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                docs.append(Document(content=content, source=f"s3://{self.bucket}/{key}", doc_id=key))
        return docs


class NotionLoader(BaseLoader):
    def __init__(self, token: str, page_id: str) -> None:
        self.token = token
        self.page_id = page_id

    def load(self) -> List[Document]:
        # Placeholder: Notion API integration requires official client or API calls.
        raise NotImplementedError("NotionLoader requires a Notion API integration; implement via official API client.")


class GDriveLoader(BaseLoader):
    def __init__(self, credentials_json: str, folder_id: str) -> None:
        self.credentials_json = credentials_json
        self.folder_id = folder_id

    def load(self) -> List[Document]:
        # Placeholder: Google Drive API integration requires google-api-python-client
        raise NotImplementedError("GDriveLoader requires Google Drive API client and OAuth credentials.")


def _strip_html(html_text: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text("\n")
    # Fallback: remove tags crudely
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()
