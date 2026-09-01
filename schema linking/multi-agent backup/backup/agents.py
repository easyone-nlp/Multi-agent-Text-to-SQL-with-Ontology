from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Protocol

from .models import AgentProposal, DatabaseSchema


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")

# Draft dictionary: intentionally small and easy to replace with embeddings or an LLM.
KO_TO_SCHEMA = {
    "가수": {"singer"},
    "콘서트": {"concert"},
    "경기": {"match", "game"},
    "선수": {"player"},
    "팀": {"team"},
    "학교": {"school", "college"},
    "학생": {"student"},
    "교수": {"teacher", "professor", "faculty"},
    "직원": {"employee", "staff"},
    "부서": {"department"},
    "회사": {"company"},
    "고객": {"customer", "client"},
    "주문": {"order"},
    "상품": {"product", "item"},
    "영화": {"movie", "film"},
    "배우": {"actor"},
    "노래": {"song"},
    "앨범": {"album"},
    "공항": {"airport"},
    "항공편": {"flight"},
    "도시": {"city"},
    "나라": {"country", "nation"},
    "국가": {"country", "nation"},
    "이름": {"name"},
    "제목": {"title", "name"},
    "나이": {"age"},
    "날짜": {"date"},
    "시간": {"time"},
    "연도": {"year"},
    "년도": {"year"},
    "주소": {"address", "location"},
    "위치": {"location", "address"},
    "가격": {"price", "cost"},
    "급여": {"salary", "pay"},
    "점수": {"score", "rating"},
    "순위": {"rank"},
    "설명": {"description"},
}

INTENT_TO_COLUMNS = {
    "누구": {"name"},
    "이름": {"name"},
    "몇 살": {"age"},
    "나이": {"age"},
    "언제": {"date", "time", "year"},
    "날짜": {"date"},
    "연도": {"year"},
    "어디": {"location", "address", "city", "country"},
    "가격": {"price", "cost"},
    "평균": {"average", "avg"},
    "최고": {"max", "score", "rating"},
    "최저": {"min", "score", "rating"},
}


class SchemaAgent(Protocol):
    name: str

    def propose(
        self,
        question: str,
        schema: DatabaseSchema,
        previous: list[AgentProposal],
    ) -> AgentProposal: ...


class LexicalScoutAgent:
    name = "lexical_scout"

    def propose(
        self,
        question: str,
        schema: DatabaseSchema,
        previous: list[AgentProposal],
    ) -> AgentProposal:
        del previous
        question_lower = question.lower()
        question_tokens = set(TOKEN_PATTERN.findall(question_lower))
        concepts = set(question_tokens)
        matched_aliases: list[str] = []
        for korean, english_terms in KO_TO_SCHEMA.items():
            if korean in question_lower:
                concepts.update(english_terms)
                matched_aliases.append(f"{korean}→{','.join(sorted(english_terms))}")

        table_scores: dict[str, float] = {}
        column_scores: dict[str, float] = {}
        for table in schema.tables:
            score = _name_overlap(table, concepts)
            if score:
                table_scores[table] = score
        for column in schema.columns:
            score = _name_overlap(column.name, concepts)
            if score:
                column_scores[column.key] = score
                table_scores[column.table] = table_scores.get(column.table, 0.0) + score * 0.35

        reasons = [f"한국어 별칭 매칭: {', '.join(matched_aliases)}"] if matched_aliases else []
        if not reasons:
            reasons.append("직접 일치하는 영문/숫자 스키마 토큰을 탐색")
        return AgentProposal(self.name, table_scores, column_scores, reasons)


class IntentScoutAgent:
    name = "intent_scout"

    def propose(
        self,
        question: str,
        schema: DatabaseSchema,
        previous: list[AgentProposal],
    ) -> AgentProposal:
        del previous
        question_lower = question.lower()
        wanted: set[str] = set()
        cues: list[str] = []
        for cue, concepts in INTENT_TO_COLUMNS.items():
            if cue in question_lower:
                wanted.update(concepts)
                cues.append(cue)

        table_scores: defaultdict[str, float] = defaultdict(float)
        column_scores: dict[str, float] = {}
        for column in schema.columns:
            score = _name_overlap(column.name, wanted)
            if score:
                column_scores[column.key] = score
                table_scores[column.table] += score * 0.4

        count_intent = any(cue in question_lower for cue in ("몇 명", "몇 개", "개수", "총 수"))
        reasons = []
        if cues:
            reasons.append(f"질의 의도 단서: {', '.join(cues)}")
        if count_intent:
            reasons.append("COUNT 계열 질의로 판단; 특정 출력 컬럼 강제 선택 안 함")
        if not reasons:
            reasons.append("명시적인 출력 컬럼 의도 단서가 없음")
        return AgentProposal(self.name, dict(table_scores), column_scores, reasons)


class SchemaGraphAgent:
    name = "schema_graph"

    def __init__(self, path_score: float = 10.0) -> None:
        self.path_score = path_score

    def propose(
        self,
        question: str,
        schema: DatabaseSchema,
        previous: list[AgentProposal],
    ) -> AgentProposal:
        del question
        llm_anchors = sorted(
            {
                table
                for proposal in previous
                if proposal.agent in {"llm_schema_scout", "llm_schema_critic", "db_value_grounder"}
                for table in proposal.table_scores
            }
        )
        if llm_anchors:
            return self._path_closure(schema, llm_anchors)
        return self._heuristic_fallback(schema, previous)

    def _path_closure(
        self, schema: DatabaseSchema, anchors: list[str]
    ) -> AgentProposal:
        adjacency: defaultdict[str, set[str]] = defaultdict(set)
        edge_columns: defaultdict[frozenset[str], list[tuple[str, str]]] = defaultdict(list)
        for left, right in schema.foreign_keys:
            left_table = left.split(".", 1)[0]
            right_table = right.split(".", 1)[0]
            adjacency[left_table].add(right_table)
            adjacency[right_table].add(left_table)
            edge_columns[frozenset((left_table, right_table))].append((left, right))

        table_scores: dict[str, float] = {}
        column_scores: dict[str, float] = {}
        paths: list[str] = []
        disconnected: list[str] = []
        for index, left_anchor in enumerate(anchors):
            for right_anchor in anchors[index + 1 :]:
                path = self._shortest_path(adjacency, left_anchor, right_anchor)
                if not path:
                    disconnected.append(f"{left_anchor}↔{right_anchor}")
                    continue
                paths.append("→".join(path))
                for table in path[1:-1]:
                    table_scores[table] = self.path_score
                for left_table, right_table in zip(path, path[1:]):
                    for left_column, right_column in edge_columns[
                        frozenset((left_table, right_table))
                    ]:
                        column_scores[left_column] = self.path_score
                        column_scores[right_column] = self.path_score

        reasons = [
            f"LLM semantic anchors={','.join(anchors)}",
            f"FK shortest paths={'; '.join(paths) or '없음'}",
        ]
        if disconnected:
            reasons.append(f"연결 경로 없음: {', '.join(disconnected)}")
        return AgentProposal(self.name, table_scores, column_scores, reasons)

    @staticmethod
    def _shortest_path(
        adjacency: dict[str, set[str]], start: str, goal: str
    ) -> list[str]:
        queue = deque([start])
        previous: dict[str, str | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor not in previous:
                    previous[neighbor] = current
                    queue.append(neighbor)
        if goal not in previous:
            return []
        path: list[str] = []
        current: str | None = goal
        while current is not None:
            path.append(current)
            current = previous[current]
        return list(reversed(path))

    def _heuristic_fallback(
        self, schema: DatabaseSchema, previous: list[AgentProposal]
    ) -> AgentProposal:
        seed_scores: defaultdict[str, float] = defaultdict(float)
        for proposal in previous:
            for table, score in proposal.table_scores.items():
                seed_scores[table] += score
        if not seed_scores:
            return AgentProposal(self.name, reasons=["그래프 확장용 seed 테이블이 없음"])

        anchor = max(seed_scores, key=seed_scores.get)
        table_scores: defaultdict[str, float] = defaultdict(float)
        column_scores: defaultdict[str, float] = defaultdict(float)
        table_scores[anchor] = 0.8
        for key in schema.primary_keys:
            if key.startswith(f"{anchor}."):
                column_scores[key] += 0.35

        neighbor_tables: set[str] = set()
        for left, right in schema.foreign_keys:
            left_table, right_table = left.split(".", 1)[0], right.split(".", 1)[0]
            if anchor not in {left_table, right_table}:
                continue
            neighbor = right_table if left_table == anchor else left_table
            neighbor_tables.add(neighbor)
            table_scores[neighbor] += 0.3
            column_scores[left] += 0.45
            column_scores[right] += 0.45

        reason = f"heuristic seed={anchor}; FK 이웃={','.join(sorted(neighbor_tables)) or '없음'}"
        return AgentProposal(self.name, dict(table_scores), dict(column_scores), [reason])

class ReviewerAgent:
    name = "reviewer"

    def __init__(
        self,
        weights: dict[str, float],
        max_tables: int,
        max_columns: int,
        relative_table_threshold: float,
        relative_column_threshold: float,
        include_primary_keys: bool,
    ) -> None:
        self.weights = weights
        self.max_tables = max_tables
        self.max_columns = max_columns
        self.relative_table_threshold = relative_table_threshold
        self.relative_column_threshold = relative_column_threshold
        self.include_primary_keys = include_primary_keys

    def decide(
        self, schema: DatabaseSchema, proposals: list[AgentProposal]
    ) -> tuple[list[str], list[str], dict[str, float], dict[str, float], AgentProposal]:
        table_scores: defaultdict[str, float] = defaultdict(float)
        column_scores: defaultdict[str, float] = defaultdict(float)
        for proposal in proposals:
            weight = self.weights.get(proposal.agent, 1.0)
            for table, score in proposal.table_scores.items():
                table_scores[table] += weight * score
            for column, score in proposal.column_scores.items():
                column_scores[column] += weight * score

        trusted_agents = {
            "llm_schema_scout",
            "llm_schema_critic",
            "db_value_grounder",
            "schema_graph",
        }
        llm_active = any(
            proposal.agent in {
                "llm_schema_scout", "llm_schema_critic", "db_value_grounder"
            }
            and proposal.table_scores
            for proposal in proposals
        )
        trusted_tables = {
            table
            for proposal in proposals
            if proposal.agent in trusted_agents
            for table in proposal.table_scores
        }
        candidate_tables = trusted_tables if llm_active else set(table_scores)
        if not candidate_tables:
            table_scores[schema.tables[0]] = 0.01
            candidate_tables = {schema.tables[0]}

        ranked_tables = sorted(
            candidate_tables, key=lambda key: (-table_scores[key], key)
        )
        best = table_scores[ranked_tables[0]]
        selected_tables = [
            table
            for table in ranked_tables
            if table_scores[table] >= best * self.relative_table_threshold
        ][: self.max_tables]

        if self.include_primary_keys:
            for key in schema.primary_keys:
                if key.split(".", 1)[0] in selected_tables:
                    column_scores[key] += 0.05

        trusted_columns = {
            column
            for proposal in proposals
            if proposal.agent in trusted_agents
            for column in proposal.column_scores
        }
        eligible_columns = {
            key: score
            for key, score in column_scores.items()
            if key.split(".", 1)[0] in selected_tables
            and (not llm_active or key in trusted_columns)
        }
        selected_columns = sorted(
            eligible_columns,
            key=lambda key: (-eligible_columns[key], key),
        )[: self.max_columns]

        rationale = AgentProposal(
            agent=self.name,
            table_scores={table: table_scores[table] for table in selected_tables},
            column_scores={column: column_scores[column] for column in selected_columns},
            reasons=[
                f"가중 합의 후 table≤{self.max_tables}, column≤{self.max_columns}로 pruning",
                f"선택 테이블: {', '.join(selected_tables)}",
            ],
        )
        return (
            selected_tables,
            selected_columns,
            dict(table_scores),
            dict(column_scores),
            rationale,
        )


def _name_overlap(name: str, concepts: set[str]) -> float:
    name_tokens = set(TOKEN_PATTERN.findall(name.lower().replace("_", " ")))
    if not name_tokens or not concepts:
        return 0.0
    exact = len(name_tokens & concepts)
    partial = sum(
        1
        for name_token in name_tokens
        for concept in concepts
        if len(concept) >= 4 and (name_token.startswith(concept) or concept.startswith(name_token))
    )
    return min(1.5, exact + partial * 0.35)
