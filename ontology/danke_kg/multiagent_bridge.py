from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .dictionary import Dictionary, MatchingDiscoveryService, build_dictionary
from .graph import KnowledgeGraph, build_knowledge_graph, steiner_forest
from .mapping import RDBMapping
from .models import KnowledgeSchema

MULTI_AGENT_DIR = Path(__file__).resolve().parents[2] / "schema linking" / "multi-agent"
if str(MULTI_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(MULTI_AGENT_DIR))

from schema_agents.agentic_orchestrator import AgenticMultiAgentSchemaLinker  # noqa: E402
from schema_agents.embedding_retriever import RetrievedSchema  # noqa: E402
from schema_agents.model_client import ChatModel  # noqa: E402
from schema_agents.models import DatabaseSchema  # noqa: E402

WORD_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


class DankeSchemaRetriever:
    """Retrieval front-end backed by the DANKE-style knowledge graph

    (dictionary matching + Steiner-tree expansion) instead of embedding
    cosine similarity. Implements the same `retrieve(query, schema) ->
    RetrievedSchema-shaped object` interface as
    schema_agents.embedding_retriever.EmbeddingSchemaRetriever, so it is a
    drop-in replacement for `self.retriever` on
    AgenticMultiAgentSchemaLinker -- see DankeAugmentedSchemaLinker below,
    which swaps it in without editing any file under `schema linking/`.
    """

    def __init__(
        self,
        knowledge_schema: KnowledgeSchema,
        mapping: RDBMapping,
        dictionary: Dictionary | None = None,
        fuzzy_cutoff: float = 0.72,
        use_fuzzy: bool = True,
        max_ngram: int = 3,
        expand_with_steiner: bool = True,
    ) -> None:
        self.knowledge_schema = knowledge_schema
        self.mapping = mapping
        self.dictionary = dictionary or build_dictionary(knowledge_schema)
        self.matching_service = MatchingDiscoveryService(self.dictionary, fuzzy_cutoff=fuzzy_cutoff)
        self.graph: KnowledgeGraph = build_knowledge_graph(knowledge_schema)
        self.use_fuzzy = use_fuzzy
        self.max_ngram = max_ngram
        self.expand_with_steiner = expand_with_steiner
        # agentic_orchestrator.py logs `self.retriever.model.model` for every
        # retriever; there is no embedding model here, so stub the same shape.
        self.model = SimpleNamespace(model=f"danke_kg:{knowledge_schema.db_id}")

    def retrieve(self, query: str, schema: DatabaseSchema) -> RetrievedSchema:
        del schema  # table/column identifiers already resolved via self.mapping
        keywords = self._candidate_phrases(query)
        matches = self.matching_service.match(keywords)
        matched = [item for item in matches if item.matched and (self.use_fuzzy or not item.fuzzy)]
        matched_classes = self.matching_service.matched_classes(matched)

        classes = set(matched_classes)
        if self.expand_with_steiner and len(matched_classes) >= 2:
            for tree in steiner_forest(self.graph, matched_classes):
                classes |= tree.nodes
        if not classes:
            # No dictionary hit at all: fall back to the whole schema rather
            # than returning an empty candidate set (mirrors the embedding
            # retriever, which always returns its top-k regardless of score).
            classes = set(self.knowledge_schema.classes)

        tables = [
            self.mapping.table_for(name) for name in sorted(classes) if name in self.mapping.class_table
        ]
        columns: list[str] = []
        for name in sorted(classes):
            for prop in self.knowledge_schema.properties_of(name):
                table, column = self.mapping.column_for(prop.name)
                key = f"{table}.{column}"
                if key not in columns:
                    columns.append(key)
        return RetrievedSchema(tables=tables, columns=columns, query=query, mode="danke_kg")

    def _candidate_phrases(self, query: str) -> list[str]:
        """Sliding-window word n-grams stand in for DANKE's LLM keyword

        extraction step (Section 5.2, "call the LLM to extract keywords"):
        cheaper, and sufficient since dictionary lookup is a fast exact/fuzzy
        match rather than free-form generation.
        """
        words = WORD_PATTERN.findall(query)
        phrases: list[str] = []
        for n in range(1, self.max_ngram + 1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if len(phrase) >= 2:
                    phrases.append(phrase)
        return list(dict.fromkeys(phrases))


class DankeAugmentedSchemaLinker(AgenticMultiAgentSchemaLinker):
    """Identical manager/schema_linker/value_linker/join_linker pipeline to

    AgenticMultiAgentSchemaLinker, with its embedding-based retriever
    swapped for DankeSchemaRetriever. This only overrides an instance
    attribute after the parent constructor runs -- no file under
    `schema linking/` is modified.
    """

    def __init__(
        self,
        config: dict[str, Any] | None,
        knowledge_schema: KnowledgeSchema,
        mapping: RDBMapping,
        chat_model: ChatModel | None = None,
        retriever_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, chat_model=chat_model)
        self.retriever = DankeSchemaRetriever(knowledge_schema, mapping, **(retriever_kwargs or {}))
