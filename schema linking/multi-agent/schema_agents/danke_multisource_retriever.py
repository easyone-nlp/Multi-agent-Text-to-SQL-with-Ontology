from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .embedding_retriever import RetrievedSchema
from .models import DatabaseSchema

WORKSPACE = Path(__file__).resolve().parents[3]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from ontology.danke_kg.mapping import RDBMapping  # noqa: E402
from ontology.danke_kg.models import KnowledgeSchema  # noqa: E402


class MultiSourceDankeRetriever:
    """Production adapter for the frozen multi-source DANKE retriever.

    The experiment implementation remains the executable reference so the
    official pipeline and retriever-only evaluation cannot silently diverge.
    Knowledge artifacts are loaded lazily per db_id.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        configured_root = Path(
            str(
                config.get(
                    "knowledge_schema_dir",
                    "ontology/versions/danke-spider-ko-v1.0.0-20260728",
                )
            )
        )
        self.knowledge_schema_dir = (
            configured_root
            if configured_root.is_absolute()
            else WORKSPACE / configured_root
        ).resolve()
        self.semantic_table_budget = max(
            1, int(config.get("semantic_table_budget", 3))
        )
        self.semantic_column_budget = max(
            1, int(config.get("semantic_column_budget", 8))
        )
        self.bridge_table_budget = max(
            0, int(config.get("bridge_table_budget", 2))
        )
        self.fuzzy_cutoff = float(config.get("fuzzy_cutoff", 0.72))
        self.max_ngram = max(1, int(config.get("max_ngram", 3)))
        self.use_synonyms = bool(config.get("use_synonyms", True))
        self.use_steiner = bool(config.get("use_steiner", True))
        self.generic_property_table_gate = bool(
            config.get("generic_property_table_gate", True)
        )
        self.generic_property_keys = [
            str(item)
            for item in config.get(
                "generic_property_keys",
                ["id", "identifier", "name", "아이디", "식별자", "이름"],
            )
        ]
        self.strict_top_k = True
        self.top_k_tables = self.semantic_table_budget
        self.top_k_columns = self.semantic_column_budget
        self.model = SimpleNamespace(model="danke_multisource_v2.1")
        self._artifact_cache: dict[
            str, tuple[KnowledgeSchema, RDBMapping, Any]
        ] = {}

    def retrieve_multisource(
        self,
        question: str,
        decomposition: dict[str, Any],
        schema: DatabaseSchema,
    ) -> RetrievedSchema:
        retrieval_query = str(
            decomposition.get("retrieval_query") or question
        )
        source = {
            "question": question,
            "retrieval_query": retrieval_query,
            "decomposition": decomposition,
        }
        return self._retrieve(source, schema, mode="multi_source")

    def retrieve(
        self, query: str, schema: DatabaseSchema
    ) -> RetrievedSchema:
        source = {
            "question": query,
            "retrieval_query": query,
            "decomposition": {},
        }
        return self._retrieve(source, schema, mode="retrieval_query")

    def _retrieve(
        self,
        source: dict[str, Any],
        schema: DatabaseSchema,
        *,
        mode: str,
    ) -> RetrievedSchema:
        # Lazy import avoids a package cycle while keeping one executable
        # retrieval implementation for the experiment and main pipeline.
        from ontology.compare_retrievers_fair import (
            _danke_query_sources,
            _danke_retrieve,
        )

        knowledge_schema, mapping, dictionary = self._artifacts(schema.db_id)
        output = _danke_retrieve(
            query_sources=_danke_query_sources(source, mode=mode),
            schema=schema,
            knowledge_schema=knowledge_schema,
            mapping=mapping,
            dictionary=dictionary,
            use_steiner=self.use_steiner,
            args=SimpleNamespace(
                fuzzy_cutoff=self.fuzzy_cutoff,
                max_ngram=self.max_ngram,
                semantic_table_budget=self.semantic_table_budget,
                semantic_column_budget=self.semantic_column_budget,
                bridge_table_budget=self.bridge_table_budget,
                generic_property_table_gate=self.generic_property_table_gate,
                generic_property_keys=self.generic_property_keys,
            ),
        )
        query = str(source.get("retrieval_query") or source.get("question") or "")
        return RetrievedSchema(
            tables=list(output["effective_tables"]),
            columns=list(output["all_columns"]),
            query=query,
            mode="danke_multisource",
            top_k_tables=self.semantic_table_budget,
            top_k_columns=self.semantic_column_budget,
            metadata={
                "semantic_tables": output["semantic_tables"],
                "semantic_columns": output["semantic_columns"],
                "bridge_tables": output["bridge_tables"],
                "bridge_candidates_before_budget": output[
                    "bridge_candidates_before_budget"
                ],
                "join_key_columns": output["join_key_columns"],
                "join_edges": output["join_edges"],
                "raw": output["raw"],
                "no_match_fallback": output["no_match_fallback"],
            },
        )

    def _artifacts(
        self, db_id: str
    ) -> tuple[KnowledgeSchema, RDBMapping, Any]:
        cached = self._artifact_cache.get(db_id)
        if cached is not None:
            return cached
        schema_path = self.knowledge_schema_dir / f"{db_id}_knowledge_schema.json"
        mapping_path = self.knowledge_schema_dir / f"{db_id}_mapping.json"
        if not schema_path.is_file() or not mapping_path.is_file():
            raise RuntimeError(
                "DANKE artifact가 없습니다: "
                f"db_id={db_id}, schema={schema_path}, mapping={mapping_path}"
            )
        knowledge_schema = KnowledgeSchema.from_dict(
            json.loads(schema_path.read_text(encoding="utf-8"))
        )
        mapping = RDBMapping.from_dict(
            json.loads(mapping_path.read_text(encoding="utf-8"))
        )
        from ontology.compare_retrievers_fair import _build_dictionary

        dictionary = _build_dictionary(
            knowledge_schema, use_synonyms=self.use_synonyms
        )
        cached = (knowledge_schema, mapping, dictionary)
        self._artifact_cache[db_id] = cached
        return cached
