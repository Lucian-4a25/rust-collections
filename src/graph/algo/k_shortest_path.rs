use std::{
    collections::{BinaryHeap, HashMap},
    hash::Hash,
};

use super::Measure;
use crate::graph::visit::EdgeRef;
use crate::graph::{algo::MinScore, visit::IntoEdges};

// k_shortest_path shortest algorithm, if target is Some(_), only calculate the shortest
// path between start and target.
pub fn k_shortest_path<G, F, K>(
    graph: G,
    source: G::NodeId,
    target: Option<G::NodeId>,
    k: usize,
    mut edge_cost: F,
) -> HashMap<G::NodeId, K>
where
    G: IntoEdges,
    G::NodeId: Hash + Eq,
    F: FnMut(&G::EdgeWeight) -> K,
    K: Measure + Default,
{
    assert!(k > 0);
    let mut priority_queue = BinaryHeap::new();
    let mut counter = HashMap::new();
    let mut result = HashMap::new();
    priority_queue.push(MinScore(K::default(), source));

    while let Some(MinScore(distance, node)) = priority_queue.pop() {
        let count = counter.entry(node).or_insert(0usize);
        *count += 1;

        if k < *count {
            continue;
        }

        if *count == k {
            result.insert(node, distance);
        }

        if target.is_some() && target == Some(node) && *count == k {
            break;
        }

        for edge in graph.edges(node) {
            priority_queue.push(MinScore(distance + edge_cost(edge.weight()), edge.target()));
        }
    }

    result
}
