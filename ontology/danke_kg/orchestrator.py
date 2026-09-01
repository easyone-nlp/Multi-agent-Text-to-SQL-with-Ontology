from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents import AgentWorkflowError, RelationshipOntologyAgent, TableOntologyAgent
from .mapping import RDBMapping, _safe_name, direct_mapping
from .model_client import ChatModel, ModelClientError, build_chat_model
from .models import KnowledgeSchema
from .relational import RelationalSchema

DEFAULT_CONFIG: dict[str, Any] = {
    "provider": "heuristic",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "base_url": "http://localhost:8000/v1",
    "api_key_env": "OPENAI_API_KEY",
    "temperature": 0.0,
    "timeout_seconds": 60,
    "json_retries": 1,
    "include_synonyms": True,
}


@dataclass
class OntologyBuildResult:
    schema: KnowledgeSchema
    mapping: RDBMapping
    trace: list[dict[str, Any]] = field(default_factory=list)


class OntologyBuilder:
    """Manager-style multi-agent pipeline that turns a relational schema

    into a DANKE-like knowledge schema S^K=(C,P,O) plus its mapping mu to
    S^R. Step 1 builds the direct-mapping skeleton (tables->classes,
    columns->datatype properties, FKs->object properties). Step 2 fans out
    one TableOntologyAgent per table and one RelationshipOntologyAgent for
    all foreign keys to enrich labels, Korean synonyms, rankings, indexed
    flags, and value synonyms -- the same "specialist agent + deterministic
    aggregation" shape used by the Text-to-SQL schema-linking pipeline.
    Without a configured chat model, the builder falls back to the direct
    mapping only (Section 5.4's "quick way to apply the strategy").
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        chat_model: ChatModel | None = None,
    ) -> None:
        merged = {**DEFAULT_CONFIG, **(config or {})}
        self.model = chat_model or build_chat_model(merged)
        retries = int(merged["json_retries"])
        self.include_synonyms = bool(merged["include_synonyms"])
        self.table_agent = TableOntologyAgent(self.model, retries) if self.model else None
        self.relationship_agent = (
            RelationshipOntologyAgent(self.model, retries) if self.model else None
        )

    def build(
        self,
        schema: RelationalSchema,
        inferred_fk_pairs: set[tuple[str, str]] | None = None,
    ) -> OntologyBuildResult:
        knowledge_schema, mapping = direct_mapping(schema, inferred_fk_pairs)
        trace: list[dict[str, Any]] = [
            {
                "agent": "direct_mapping",
                "status": "ok",
                "task": "build class/property/object-property skeleton from S^R",
                "output": {
                    "classes": len(knowledge_schema.classes),
                    "datatype_properties": len(knowledge_schema.datatype_properties),
                    "object_properties": len(knowledge_schema.object_properties),
                },
            }
        ]

        if self.table_agent is None:
            trace.append(
                {
                    "agent": "table_ontology_agent",
                    "status": "skipped",
                    "task": "no chat model configured; keeping direct-mapping labels",
                }
            )
        else:
            for table in schema.tables:
                self._enrich_table(schema, knowledge_schema, mapping, table, trace)

        if self.relationship_agent is None or not knowledge_schema.object_properties:
            trace.append(
                {
                    "agent": "relationship_ontology_agent",
                    "status": "skipped",
                    "task": "no chat model configured or no foreign keys to label",
                }
            )
        else:
            self._enrich_relationships(knowledge_schema, mapping, trace)

        return OntologyBuildResult(schema=knowledge_schema, mapping=mapping, trace=trace)

    def _enrich_table(
        self,
        schema: RelationalSchema,
        knowledge_schema: KnowledgeSchema,
        mapping: RDBMapping,
        table: str,
        trace: list[dict[str, Any]],
    ) -> None:
        class_name = next(
            (name for name, source in mapping.class_table.items() if source == table),
            None,
        )
        if class_name is None:
            return
        columns = schema.columns_for(table)
        assert self.table_agent is not None
        korean_table = schema.korean_tables.get(table)
        korean_columns = {
            column.name: schema.korean_columns[column.key]
            for column in columns
            if column.key in schema.korean_columns
        }
        try:
            response = self.table_agent.enrich_table(
                table,
                columns,
                schema.primary_keys,
                schema.foreign_keys,
                korean_table=korean_table,
                korean_columns=korean_columns,
                include_synonyms=self.include_synonyms,
            )
        except (AgentWorkflowError, ModelClientError) as error:
            trace.append(
                {
                    "agent": "table_ontology_agent",
                    "status": "error",
                    "task": table,
                    "error": str(error),
                }
            )
            return

        payload = response.payload
        knowledge_class = knowledge_schema.classes[class_name]
        authentic_class_label = knowledge_class.primary_label
        class_out = payload.get("class", {})
        if class_out.get("primary_label"):
            knowledge_class.primary_label = str(class_out["primary_label"])
        knowledge_class.synonyms = list(class_out.get("synonyms", []))
        knowledge_class.ranking = float(class_out.get("ranking", knowledge_class.ranking))
        if self.include_synonyms:
            _preserve_authentic_label(knowledge_class, authentic_class_label)

        by_column = {column.name: column for column in columns}
        for item in payload.get("properties", []):
            column = by_column.get(item["column"])
            if column is None:
                continue
            prop_name = f"{class_name}_{_safe_name(column.name)}"
            prop = knowledge_schema.datatype_properties.get(prop_name)
            if prop is None:
                continue
            authentic_property_label = prop.primary_label
            if item.get("primary_label"):
                prop.primary_label = str(item["primary_label"])
            prop.synonyms = list(item.get("synonyms", []))
            prop.ranking = float(item.get("ranking", prop.ranking))
            prop.indexed = bool(item.get("indexed", prop.indexed))
            prop.value_synonyms = dict(item.get("value_synonyms", {}))
            if self.include_synonyms:
                _preserve_authentic_label(prop, authentic_property_label)

        trace.append(
            {
                "agent": "table_ontology_agent",
                "status": "ok",
                "task": table,
                "attempts": response.attempts,
                "output": payload,
            }
        )

    def _enrich_relationships(
        self,
        knowledge_schema: KnowledgeSchema,
        mapping: RDBMapping,
        trace: list[dict[str, Any]],
    ) -> None:
        relationships = []
        by_pair: dict[tuple[str, str], str] = {}
        for op_name, (left, right) in mapping.object_property_fk.items():
            prop = knowledge_schema.object_properties[op_name]
            relationships.append(
                {
                    "left": left,
                    "right": right,
                    "left_class": prop.domain,
                    "right_class": prop.range,
                    "inferred": prop.inferred,
                }
            )
            by_pair[(left, right)] = op_name

        assert self.relationship_agent is not None
        try:
            response = self.relationship_agent.enrich_relationships(
                relationships, include_synonyms=self.include_synonyms
            )
        except (AgentWorkflowError, ModelClientError) as error:
            trace.append(
                {
                    "agent": "relationship_ontology_agent",
                    "status": "error",
                    "task": "label foreign keys",
                    "error": str(error),
                }
            )
            return

        for item in response.payload.get("object_properties", []):
            op_name = by_pair.get((item["left"], item["right"]))
            if op_name is None:
                continue
            prop = knowledge_schema.object_properties[op_name]
            if item.get("primary_label"):
                prop.primary_label = str(item["primary_label"])
            prop.synonyms = list(item.get("synonyms", []))
            prop.weight = float(item.get("weight", prop.weight))

        trace.append(
            {
                "agent": "relationship_ontology_agent",
                "status": "ok",
                "task": "label foreign keys",
                "attempts": response.attempts,
                "output": response.payload,
            }
        )


def _preserve_authentic_label(entity: Any, authentic_label: str) -> None:
    """Keep the source schema's own (often Korean) name as a synonym even if

    the LLM chose a different primary_label -- it should never be lost.
    """
    if not authentic_label or authentic_label == entity.primary_label:
        return
    if authentic_label not in entity.synonyms:
        entity.synonyms.append(authentic_label)

