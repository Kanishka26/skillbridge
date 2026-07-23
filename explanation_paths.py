"""
Relational path extraction for explainability.

Given a predicted missing skill and the target occupation, finds the
shortest path through the knowledge graph connecting them, so the API
can explain WHY a skill was predicted as missing (e.g. it's required for
a different skill that's required for the target occupation).

Uses BFS over the NetworkX graph G (built in skillbridge_full_rebuild.py),
NOT the PyG HeteroData -- G retains human-readable node labels and edge
type strings, which is what a path explanation actually needs to display.
"""
import networkx as nx


def format_edge_label(edge_type):
    """Map internal edge_type strings to readable relation phrases."""
    mapping = {
        'essential': 'is required for',
        'optional': 'is optionally useful for',
        'onet_software': 'is used in',
    }
    return mapping.get(edge_type, edge_type)


EDGE_STRENGTH = {'essential': 0, 'onet_software': 1, 'optional': 2}  # lower = stronger, preferred

def find_explanation_path(G, skill_uri, occupation_uri, max_hops=4):
    """
    Returns the shortest path (as a list of (node, edge_label) hops) from
    skill_uri to occupation_uri, or None if no path exists within max_hops.

    Among all shortest paths of equal length, prefers the one using the
    strongest relation types (essential > onet_software > optional) at
    each hop, since a path built from 'essential' edges is a more
    convincing explanation than one built from weaker 'optional' links.
    """
    if skill_uri not in G or occupation_uri not in G:
        return None

    G_undirected = nx.Graph(G)

    try:
        shortest_len = nx.shortest_path_length(G_undirected, source=skill_uri,
                                                 target=occupation_uri)
    except nx.NetworkXNoPath:
        return None

    if shortest_len > max_hops:
        return None

    # Collect ALL shortest paths (not just one), then pick the strongest.
    all_shortest = list(nx.all_shortest_paths(G_undirected, source=skill_uri,
                                                target=occupation_uri))

    def path_strength_score(path_nodes):
        total = 0
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            edge_type = _get_edge_type(G, u, v)
            total += EDGE_STRENGTH.get(edge_type, 3)  # unknown types treated as weakest
        return total

    best_path = min(all_shortest, key=path_strength_score)

    hops = []
    for i in range(len(best_path) - 1):
        u, v = best_path[i], best_path[i + 1]
        edge_type = _get_edge_type(G, u, v)
        hops.append({
            'from': G.nodes[u].get('label', u),
            'relation': format_edge_label(edge_type),
            'to': G.nodes[v].get('label', v),
        })
    return hops


def _get_edge_type(G, u, v):
    if G.has_edge(u, v):
        edge_data = G.get_edge_data(u, v)
        return list(edge_data.values())[0].get('edge_type', 'related to')
    elif G.has_edge(v, u):
        edge_data = G.get_edge_data(v, u)
        return list(edge_data.values())[0].get('edge_type', 'related to')
    return 'related to'


def format_path_as_string(hops):
    """Turns hop list into 'A -[relation]-> B -[relation]-> C' display string."""
    if not hops:
        return None
    parts = [hops[0]['from']]
    for hop in hops:
        parts.append(f"-[{hop['relation']}]-> {hop['to']}")
    return " ".join(parts)


def format_path_short(hops):
    """
    Compact version for UI display: 'A ... -> C' style, showing only the
    first and last node plus hop count, rather than the full chain.
    Useful for cards/lists where the full path is too long but the UI
    still wants to show something rather than nothing.
    """
    if not hops:
        return None
    if len(hops) == 1:
        return f"{hops[0]['from']} -[{hops[0]['relation']}]-> {hops[0]['to']}"
    return f"{hops[0]['from']} -- {len(hops)} steps --> {hops[-1]['to']}"


# ── Example usage / test ──────────────────────────────────────
if __name__ == "__main__":
    # Run in the same kernel session as skillbridge_full_rebuild.py (needs G,
    # all_skill_nodes, all_occ_nodes).
    import random
    random.seed(0)

    sample_skill = random.choice(all_skill_nodes)
    sample_occ   = random.choice(all_occ_nodes)

    print(f"Finding path: {G.nodes[sample_skill]['label']} -> {G.nodes[sample_occ]['label']}")
    hops = find_explanation_path(G, sample_skill, sample_occ)
    if hops:
        print(format_path_as_string(hops))
    else:
        print("No path found within max_hops.")