"""Target-local integration which never interprets code identities as PageStore IDs."""

from __future__ import annotations

from jitmind.schemas.search import Hit

from .models import CodeQuery


class CodeContextRetriever:
    name = "code_context"

    def __init__(self, service, scope, repo_id, snapshot_id, *, path_prefix=""):
        self.service = service
        self.scope = scope
        self.repo_id = repo_id
        self.snapshot_id = snapshot_id
        self.path_prefix = path_prefix
        self.last_pages = ()

    def search(self, query_list: list[str], top_k: int = 10) -> list[list[Hit]]:
        self.service.authority.require(self.scope, self.repo_id)
        if type(query_list) is not list or len(query_list) > 16:
            raise ValueError("Query batch budget")
        pages = []
        hits = []
        for text in query_list:
            page = self.service.query(
                self.scope,
                CodeQuery(
                    operation="find_code",
                    repo_id=self.repo_id,
                    snapshot_id=self.snapshot_id,
                    query=text,
                    path_prefix=self.path_prefix,
                    limit=top_k,
                ),
            )
            pages.append(page)
            hits.append(
                [
                    Hit(
                        page_id=None,
                        snippet=e.source,
                        source="code_context",
                        meta={
                            "code_evidence": e.model_dump(),
                            "generation": page.generation,
                            "coverage": page.coverage,
                            "freshness": page.freshness,
                            "reason_codes": page.reason_codes,
                        },
                    )
                    for e in page.results
                ]
            )
        self.service.authority.require(self.scope, self.repo_id)
        self.last_pages = tuple(pages)
        return hits
