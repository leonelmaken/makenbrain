"""
Graphe de neurones — Phase 3
Chaque concept = un nœud. Chaque relation = une arête pondérée.
Le cerveau connecte automatiquement ses propres connaissances.
"""
import json
from pathlib import Path
from datetime import datetime

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

GRAPH_PATH = Path("brain_data/neuron_graph.json")


class NeuronGraph:
    def __init__(self):
        if not NX_AVAILABLE:
            raise RuntimeError("networkx non installé — lance : pip install networkx")
        self.G = nx.Graph()
        self._load()

    def _load(self):
        if not GRAPH_PATH.exists():
            return
        try:
            data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
            for name, attrs in data.get("nodes", {}).items():
                self.G.add_node(name, **attrs)
            for e in data.get("edges", []):
                self.G.add_edge(
                    e["source"], e["target"],
                    weight=e.get("weight", 1.0),
                    relationship=e.get("relationship", "related"),
                    co_occurrences=e.get("co_occurrences", 1),
                )
        except Exception:
            pass

    def save(self):
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.now().isoformat(),
            "nodes": {n: dict(self.G.nodes[n]) for n in self.G.nodes},
            "edges": [
                {"source": u, "target": v, **self.G.edges[u, v]}
                for u, v in self.G.edges
            ],
        }
        GRAPH_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_concept(self, name: str, concept_type: str = "concept", memory_id: str = ""):
        name = name.lower().strip()
        if len(name) < 3:
            return
        if name in self.G:
            self.G.nodes[name]["frequency"] = self.G.nodes[name].get("frequency", 0) + 1
            if memory_id:
                ids = self.G.nodes[name].get("memory_ids", [])
                if memory_id not in ids:
                    ids.append(memory_id)
                self.G.nodes[name]["memory_ids"] = ids
        else:
            self.G.add_node(name, type=concept_type, frequency=1,
                            memory_ids=[memory_id] if memory_id else [],
                            created_at=datetime.now().isoformat())

    def add_connection(self, a: str, b: str, relationship: str = "related", weight: float = 1.0):
        a, b = a.lower().strip(), b.lower().strip()
        if a == b or len(a) < 3 or len(b) < 3:
            return
        if a not in self.G:
            self.add_concept(a)
        if b not in self.G:
            self.add_concept(b)
        if self.G.has_edge(a, b):
            self.G.edges[a, b]["co_occurrences"] = self.G.edges[a, b].get("co_occurrences", 1) + 1
            self.G.edges[a, b]["weight"] = min(1.0, self.G.edges[a, b]["weight"] + 0.05)
        else:
            self.G.add_edge(a, b, relationship=relationship,
                            weight=round(weight, 3), co_occurrences=1)

    def explore(self, concept: str, depth: int = 2) -> dict:
        concept = concept.lower().strip()
        if concept not in self.G:
            candidates = [n for n in self.G.nodes if concept in n or n in concept]
            return {"found": False, "concept": concept, "suggestions": candidates[:5]}

        neighbors = {}
        for target, path in nx.single_source_shortest_path(self.G, concept, cutoff=depth).items():
            if target == concept:
                continue
            d = len(path) - 1
            edge = self.G.edges.get((concept, target), self.G.edges.get((target, concept), {}))
            neighbors[target] = {
                "distance": d,
                "relationship": edge.get("relationship", "indirect"),
                "strength": round(edge.get("weight", 0.5), 3),
                "path": " → ".join(path),
            }

        sorted_n = dict(sorted(
            neighbors.items(),
            key=lambda x: (x[1]["distance"], -x[1]["strength"])
        )[:25])

        return {
            "found": True,
            "concept": concept,
            "type": self.G.nodes[concept].get("type", "concept"),
            "frequency": self.G.nodes[concept].get("frequency", 1),
            "direct_connections": self.G.degree(concept),
            "neighbors": sorted_n,
        }

    def find_path(self, a: str, b: str) -> dict:
        a, b = a.lower().strip(), b.lower().strip()
        try:
            path = nx.shortest_path(self.G, a, b, weight=None)
            edges_info = []
            for i in range(len(path) - 1):
                e = self.G.edges.get((path[i], path[i+1]), {})
                edges_info.append(e.get("relationship", "related"))
            return {"found": True, "path": path, "relationships": edges_info,
                    "hops": len(path)-1, "chain": " → ".join(path)}
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"found": False, "from": a, "to": b, "reason": "aucun chemin"}

    def get_stats(self) -> dict:
        n = self.G.number_of_nodes()
        if n == 0:
            return {"nodes": 0, "edges": 0, "density": 0.0, "top_concepts": []}
        dc  = nx.degree_centrality(self.G)
        top = sorted(dc.items(), key=lambda x: -x[1])[:12]
        return {
            "nodes": n,
            "edges": self.G.number_of_edges(),
            "density": round(nx.density(self.G), 4),
            "components": nx.number_connected_components(self.G),
            "top_concepts": [
                {"concept": c, "centrality": round(v, 3),
                 "connections": self.G.degree(c),
                 "type": self.G.nodes[c].get("type", "concept")}
                for c, v in top
            ],
        }

    def get_map(self, max_nodes: int = 80) -> dict:
        all_nodes = sorted(self.G.nodes(data=True),
                           key=lambda x: -x[1].get("frequency", 0))[:max_nodes]
        node_set  = {n for n, _ in all_nodes}
        nodes = [{"id": n, "type": d.get("type","concept"),
                  "frequency": d.get("frequency",1),
                  "connections": self.G.degree(n)} for n, d in all_nodes]
        edges = [{"source": u, "target": v,
                  "weight": round(self.G.edges[u,v].get("weight",1.0),3),
                  "relationship": self.G.edges[u,v].get("relationship","related")}
                 for u, v in self.G.edges if u in node_set and v in node_set]
        return {"nodes": nodes, "edges": edges, "total_nodes": self.G.number_of_nodes()}

    def search_concepts(self, query: str, limit: int = 10) -> list[dict]:
        query = query.lower().strip()
        results = []
        for n, d in self.G.nodes(data=True):
            if query in n:
                results.append({"concept": n, "type": d.get("type","concept"),
                                 "frequency": d.get("frequency",1),
                                 "connections": self.G.degree(n)})
        return sorted(results, key=lambda x: -x["frequency"])[:limit]


# Singleton global
neuron_graph = NeuronGraph()
