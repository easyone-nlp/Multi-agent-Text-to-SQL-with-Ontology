from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .graph import KnowledgeGraph, SteinerTree, build_knowledge_graph, steiner_forest
from .mapping import RDBMapping
from .models import KnowledgeSchema


@dataclass
class SynthesizedView:
    """The view V DANKE synthesizes from a Steiner tree (Section 4.1):

    FROM/JOIN clauses that connect the covered classes' tables, and one
    output column per covered datatype property named "c_p" (class name
    concatenated with the property name), which is exactly how this
    package names DatatypeProperty.name (see mapping.direct_mapping).
    """

    name: str
    covered_classes: list[str]
    column_map: dict[str, str]
    sql: str
    steiner_tree: SteinerTree
    disconnected_classes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "covered_classes": self.covered_classes,
            "column_map": dict(self.column_map),
            "sql": self.sql,
            "steiner_tree": self.steiner_tree.to_dict(),
            "disconnected_classes": self.disconnected_classes,
        }


class ViewSynthesisService:
    """DANKE's View Synthesis Service: S' (classes) -> V (Section 3.2/4.1)."""

    def __init__(
        self,
        schema: KnowledgeSchema,
        mapping: RDBMapping,
        graph: KnowledgeGraph | None = None,
    ) -> None:
        self.schema = schema
        self.mapping = mapping
        self.graph = graph or build_knowledge_graph(schema)

    def synthesize(
        self, classes: list[str], only_indexed_columns: bool = False
    ) -> SynthesizedView:
        unique_classes = [name for name in dict.fromkeys(classes) if name in self.schema.classes]
        if not unique_classes:
            raise ValueError("view synthesis에 유효한 class가 없습니다.")

        forest = steiner_forest(self.graph, unique_classes)
        primary = max(forest, key=lambda tree: len(tree.nodes))
        disconnected = sorted(set(unique_classes) - primary.nodes)

        covered = sorted(primary.nodes)
        view_name = "view_" + "_".join(covered)

        select_parts: list[str] = []
        column_map: dict[str, str] = {}
        for class_name in covered:
            table = self.mapping.table_for(class_name)
            for prop in self.schema.properties_of(class_name):
                if only_indexed_columns and not prop.indexed:
                    continue
                _, column = self.mapping.column_for(prop.name)
                select_parts.append(f'  t_{class_name}."{column}" AS "{prop.name}"')
                column_map[prop.name] = f"{table}.{column}"

        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for left, right, op_name, _weight in primary.edges:
            adjacency[left].append((right, op_name))
            adjacency[right].append((left, op_name))

        root = covered[0]
        table = self.mapping.table_for(root)
        from_clause = f'FROM "{table}" AS t_{root}'
        join_clauses: list[str] = []
        visited = {root}
        queue: deque[str] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor, op_name in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                join_clauses.append(self._join_clause(op_name, neighbor))
                queue.append(neighbor)

        sql = "\n".join(
            [
                f'CREATE VIEW "{view_name}" AS',
                "SELECT",
                ",\n".join(select_parts) if select_parts else "  *",
                from_clause,
                *join_clauses,
                ";",
            ]
        )
        return SynthesizedView(
            name=view_name,
            covered_classes=covered,
            column_map=column_map,
            sql=sql,
            steiner_tree=primary,
            disconnected_classes=disconnected,
        )

    def _join_clause(self, op_name: str, neighbor_class: str) -> str:
        """Join `neighbor_class`'s table onto whichever endpoint is already

        in scope. `mapping.fk_for` always returns (domain-side, range-side)
        columns, so the ON condition is direction-independent; only the
        newly-joined table depends on which endpoint the BFS just reached.
        """
        prop = self.schema.object_properties[op_name]
        left, right = self.mapping.fk_for(op_name)
        _, left_column = left.split(".", 1)
        _, right_column = right.split(".", 1)
        on_condition = f't_{prop.domain}."{left_column}" = t_{prop.range}."{right_column}"'
        target_table = self.mapping.table_for(neighbor_class)
        return f'JOIN "{target_table}" AS t_{neighbor_class} ON {on_condition}'
