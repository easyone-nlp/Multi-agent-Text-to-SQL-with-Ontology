from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .models import KnowledgeSchema


@dataclass(frozen=True)
class ClassEdge:
    target: str
    weight: float
    object_property: str


@dataclass
class KnowledgeGraph:
    """The induced knowledge graph G^K=(V,E), V=C union P (Section 3.1).

    Object properties do not appear as nodes: (a,b) in E directly connects
    the domain/range classes of an object property. Datatype properties are
    leaf nodes attached to their domain class. Only the class-class edges
    (`adjacency`) participate in Steiner-tree matching optimization; the
    datatype-property leaves (`class_properties`) are exposed separately.
    """

    class_nodes: set[str] = field(default_factory=set)
    property_nodes: set[str] = field(default_factory=set)
    adjacency: dict[str, list[ClassEdge]] = field(default_factory=dict)
    class_properties: dict[str, list[str]] = field(default_factory=dict)


def build_knowledge_graph(schema: KnowledgeSchema) -> KnowledgeGraph:
    graph = KnowledgeGraph(
        class_nodes=set(schema.classes),
        property_nodes=set(schema.datatype_properties),
        adjacency={name: [] for name in schema.classes},
        class_properties={name: [] for name in schema.classes},
    )
    for prop in schema.datatype_properties.values():
        graph.class_properties.setdefault(prop.domain, []).append(prop.name)
    for op in schema.object_properties.values():
        if op.domain not in graph.adjacency or op.range not in graph.adjacency:
            continue
        graph.adjacency[op.domain].append(ClassEdge(op.range, op.weight, op.name))
        graph.adjacency[op.range].append(ClassEdge(op.domain, op.weight, op.name))
    return graph


@dataclass
class PathResult:
    nodes: list[str]
    edges: list[tuple[str, str, ClassEdge]]
    total_weight: float


def shortest_path(graph: KnowledgeGraph, start: str, goal: str) -> PathResult | None:
    """Dijkstra shortest path between two classes over object-property edges."""
    if start == goal:
        return PathResult([start], [], 0.0)
    distances: dict[str, float] = {start: 0.0}
    previous: dict[str, tuple[str, ClassEdge]] = {}
    visited: set[str] = set()
    heap: list[tuple[float, str]] = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        for edge in graph.adjacency.get(node, []):
            candidate = distance + edge.weight
            if candidate < distances.get(edge.target, math.inf):
                distances[edge.target] = candidate
                previous[edge.target] = (node, edge)
                heapq.heappush(heap, (candidate, edge.target))

    if goal not in distances:
        return None
    nodes = [goal]
    edges: list[tuple[str, str, ClassEdge]] = []
    current = goal
    while current != start:
        parent, edge = previous[current]
        edges.append((parent, current, edge))
        current = parent
        nodes.append(current)
    nodes.reverse()
    edges.reverse()
    return PathResult(nodes, edges, distances[goal])


def connected_components(graph: KnowledgeGraph) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in graph.class_nodes:
        if start in seen:
            continue
        component = {start}
        queue = [start]
        seen.add(start)
        while queue:
            node = queue.pop()
            for edge in graph.adjacency.get(node, []):
                if edge.target not in seen:
                    seen.add(edge.target)
                    component.add(edge.target)
                    queue.append(edge.target)
        components.append(component)
    return components


@dataclass
class SteinerTree:
    """A (approximate) minimum Steiner tree connecting a set of terminal classes."""

    terminals: set[str]
    nodes: set[str]
    edges: list[tuple[str, str, str, float]]  # (left, right, object_property, weight)
    total_weight: float
    disconnected: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminals": sorted(self.terminals),
            "covered_classes": sorted(self.nodes),
            "edges": [
                {"left": left, "right": right, "object_property": op, "weight": round(weight, 4)}
                for left, right, op, weight in self.edges
            ],
            "total_weight": round(self.total_weight, 4),
            "disconnected": sorted(self.disconnected),
        }


def steiner_forest(graph: KnowledgeGraph, terminals: list[str]) -> list[SteinerTree]:
    """Approximate Steiner-tree-per-component matching optimization (Section 3.2).

    Uses the classic Kou-Markowsky-Berman 2-approximation: shortest paths
    between all terminal pairs -> MST over that metric closure -> expand
    back to real edges -> MST again to drop redundant overlap. If the
    matched classes are not all connected, this returns one tree per
    connected component -- a Steiner forest, as the paper notes may be
    necessary when G_S is disconnected (Section 3.2).
    """
    unique_terminals = [name for name in dict.fromkeys(terminals) if name in graph.class_nodes]
    if not unique_terminals:
        return []

    component_index: dict[str, int] = {}
    for index, component in enumerate(connected_components(graph)):
        for node in component:
            component_index[node] = index

    groups: dict[int, list[str]] = defaultdict(list)
    for terminal in unique_terminals:
        groups[component_index[terminal]].append(terminal)

    return [_steiner_tree_single_component(graph, group) for group in groups.values()]


def _steiner_tree_single_component(graph: KnowledgeGraph, terminals: list[str]) -> SteinerTree:
    terminal_set = set(terminals)
    if len(terminals) == 1:
        return SteinerTree(terminal_set, set(terminals), [], 0.0)

    pair_paths: dict[tuple[str, str], PathResult] = {}
    for i, left in enumerate(terminals):
        for right in terminals[i + 1 :]:
            path = shortest_path(graph, left, right)
            if path is not None:
                pair_paths[(left, right)] = path

    metric_edges = [
        (path.total_weight, left, right) for (left, right), path in pair_paths.items()
    ]
    mst_pairs = _mst_kruskal(terminals, metric_edges)

    union_edges: dict[frozenset[str], tuple[str, str, str, float]] = {}
    union_nodes: set[str] = set(terminals)
    for left, right in mst_pairs:
        path = pair_paths.get((left, right)) or pair_paths.get((right, left))
        assert path is not None
        for edge_left, edge_right, edge in path.edges:
            key = frozenset((edge_left, edge_right))
            union_edges[key] = (edge_left, edge_right, edge.object_property, edge.weight)
            union_nodes.add(edge_left)
            union_nodes.add(edge_right)

    final_edges = _mst_kruskal_weighted(union_nodes, list(union_edges.values()))
    total_weight = sum(weight for *_rest, weight in final_edges)
    return SteinerTree(terminal_set, union_nodes, final_edges, total_weight)


def _mst_kruskal(nodes: list[str], edges: list[tuple[float, str, str]]) -> list[tuple[str, str]]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        parent[root_a] = root_b
        return True

    result: list[tuple[str, str]] = []
    for weight, left, right in sorted(edges, key=lambda item: item[0]):
        if union(left, right):
            result.append((left, right))
    return result


def _mst_kruskal_weighted(
    nodes: set[str], edges: list[tuple[str, str, str, float]]
) -> list[tuple[str, str, str, float]]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        parent[root_a] = root_b
        return True

    result: list[tuple[str, str, str, float]] = []
    for left, right, op_name, weight in sorted(edges, key=lambda item: item[3]):
        if union(left, right):
            result.append((left, right, op_name, weight))
    return result
