use crate::graph::visit::{EdgeRef, IntoEdges, Visitable};
use std::{
    collections::{hash_map::Entry, BinaryHeap, HashMap, VecDeque},
    hash::Hash,
};

use super::{Measure, MinScore};

#[allow(dead_code)]
/// A* is a more advanced algorithm based dijkstra algorithm, because it use
/// a heuristic method to determine the shortest possible path from current node
/// to desitination node; The estimate score is based from source node to current node,
/// and current node to desitination node' weight. By this way, we could better decide
/// which node to search instead of only according the shortest distance from source node.
pub fn astar<G, F, E, M, V>(
    graph: G,
    start: G::NodeId,
    goal: F,
    edge_cost: E,
    estimate_cost: M,
) -> Option<(V, Vec<G::NodeId>)>
where
    G: IntoEdges + Visitable,
    G::NodeId: Hash + Eq,
    F: Fn(G::NodeId) -> bool,
    E: Fn(G::EdgeRef) -> V,
    M: Fn(G::NodeId) -> V,
    V: Measure + Hash,
{
    let mut path_trakcer: _ = HashMap::new();
    let mut priority_queue = BinaryHeap::new();
    // record the known shortest path weight
    let mut scores_record = HashMap::new();
    // record the known estimated path weight
    let mut estimated_scores: _ = HashMap::new();
    // keep consistent with priority queue logic
    priority_queue.push(MinScore(estimate_cost(start), start));
    scores_record.insert(start, V::default());

    while let Some(MinScore(estimate_score, node)) = priority_queue.pop() {
        if goal(node) {
            let mut paths = VecDeque::new();
            paths.push_front(node);
            let mut current = node;
            while current != start {
                let last = path_trakcer.remove(&current).unwrap();
                current = last;
                paths.push_front(last);
            }
            return Some((
                scores_record.remove(&node).unwrap(),
                paths.into_iter().collect(),
            ));
        }

        // optimize: we could skip to iterate child nodes if we find there is closer path
        // to current node before
        match estimated_scores.entry(node) {
            Entry::Vacant(entry) => {
                entry.insert(estimate_score);
            }
            // There are two cases in this condition:
            // 1. For case we already search the node before, but we maybe search this
            // node from different path, and get a estimate score less than before ones.
            // In this case we may store greater equal than two esitmate value in the priority queue,
            // If current estimated is large, it indicates we already search this node from
            // a shorter path, we could skip this path immediately.
            // We could compare the estimated value to skip the larger value to save computation cost.
            // 2. Another case is, if we searched from a longer path, after that we may
            // find a shorter path to same node, so it's possible for us to update a less estimate value.
            // A example for these two case is following:
            //            E
            //            | (20 + 0)
            //            D
            //  (1 + 0) /  \ (2 + 0)
            //        B    C
            // (3 + 5) \ / (3 + 2)
            //          A
            // the order should be: A -> C -> D -> B -> D -> E,
            // first D's estimate value should be 5, then from path B -> D, the value become 4.
            // and for E, the first value should be 25 in the queue, then a less value is 24,
            // so we could skip the larger value.
            Entry::Occupied(entry) => {
                if &estimate_score >= entry.get() {
                    continue;
                }
                *entry.into_mut() = estimate_score;
            }
        }

        let node_score = scores_record[&node];
        for edge_ref in graph.edges(node) {
            let target = edge_ref.target();
            let target_score = edge_cost(edge_ref) + node_score;

            // every time we fetch from one node, we need to update all shortest path record
            match scores_record.entry(target) {
                Entry::Occupied(entry) => {
                    // there is already a closer path from source to target, we could
                    // abandon this path
                    if target_score >= *entry.get() {
                        continue;
                    }
                    *entry.into_mut() = target_score;
                }
                Entry::Vacant(entry) => {
                    entry.insert(target_score);
                }
            }

            path_trakcer.insert(target, node);

            // And the we update the estimate records, to fetch next node, we must trust
            // the estimate function is consistent, or we may get unexpected error.
            let target_estimate_score = target_score + estimate_cost(target);

            priority_queue.push(MinScore(target_estimate_score, target));
        }
    }

    None
}

#[cfg(test)]
mod test_astar {
    use super::*;

    // Graph represented with the weight of each edge
    // Edges with '*' are part of the optimal path.
    //
    //     2       1
    // a ----- b ----- c
    // | 4*    | 7     |
    // d       f       | 5
    // | 1*    | 1*    |
    // \------ e ------/
    #[test]
    fn test_basic() {
        use crate::graph::graph_adjacency_list::Graph;
        let mut g = Graph::new();
        let a = g.add_node((0., 0.));
        let b = g.add_node((2., 0.));
        let c = g.add_node((1., 1.));
        let d = g.add_node((0., 2.));
        let e = g.add_node((3., 3.));
        let f = g.add_node((4., 2.));
        for edge in [
            (a, b, 2),
            (a, d, 4),
            (b, c, 1),
            (b, f, 7),
            (c, e, 5),
            (e, f, 1),
            (d, e, 1),
        ] {
            g.add_edge(edge.0, edge.1, edge.2);
        }

        let path = astar(&g, a, |finish| finish == f, |e| *e.weight(), |_| 0);
        assert_eq!(path, Some((6, vec![a, d, e, f])));
    }
}
