use std::{
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
};

use crate::{
    binary_heap_cus::BinaryHeap,
    graph::visit::{EdgeRef, IntoEdges, VisitMap, Visitable},
};

use super::{Measure, MinScore};

// What will you do to implement a shortest path algorithem?
// 1. find all neighbor nodes, and record the distance between current node with them,
// make the key value is the target node, the value is the distance.
// 2. iterate all last group of nodes, get all their neighbors nodes, and caculate the distanece
// of their neighbors with source node (the distance value is their distance plus the distance
// between them and the their neighbors), if the new nodes is already in the Map, compare the
// distance value, if less than current, update the value, or we abandon it.
// 3. recurse do step 2, until we find a path from source to the target, and we mark current distance
// value; If we could find a new neigbors which has a value less than it, we update it.
// or if its distance value is greater than current value, we could abandon that node's neighbor nodes.
// 4. Do step 2 and 3 recursely, until we find all neighbors nodes' distance value is greater than
// current value or we have no new neighbors, then the current value is the shortest distance value.

// Analysis:
// The disadvantage of this method is that we iterate all the node and its neighbors in the path,
// that's a big cost of computation. Is there a better way to solve this problem?
// Answer:
// Compare with all kinds of algorithm to find answers.

#[allow(dead_code)]
/// A more efficent path search algorithm than DFS or BFS algorithm, it always try to use
/// closest path first to find target nodes.
pub fn dijkstra<G, F, V>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    edge_cost: F,
) -> HashMap<G::NodeId, V>
where
    G: IntoEdges + Visitable,
    G::NodeId: PartialEq + Eq + Hash,
    F: Fn(G::EdgeRef) -> V,
    V: Measure,
{
    let mut priority_queue = BinaryHeap::new();
    let mut visited = graph.visit_map();
    let mut result: _ = HashMap::new();
    priority_queue.push(MinScore(V::default(), start));

    // always try to visit the closest point
    while let Some(MinScore(score, node)) = priority_queue.pop() {
        // we may add multiple possible path to one node before we visit it.
        if visited.is_visit(node) {
            continue;
        }
        visited.visit(node);

        // record the path value when we visit it
        match result.entry(node) {
            Entry::Occupied(_) => {
                // this is impossible behavior, if this condition matches, there is mistake
                // in our algorithm implemention
                unreachable!();
            }
            Entry::Vacant(entry) => {
                entry.insert(score);
            }
        }

        if goal.map(|g| g == node).unwrap_or(false) {
            break;
        }

        let edge_refs = graph.edges(node);
        for edge_ref in edge_refs {
            let target = edge_ref.target();
            if !visited.is_visit(target) {
                priority_queue.push(MinScore(edge_cost(edge_ref) + score, target));
            }
        }
    }

    result
}

#[cfg(test)]
mod test_dijkstra {
    use super::*;

    // a ----> b ----> e ----> f
    // ^       |       ^       |
    // |       v       |       v
    // d <---- c       h <---- g
    #[test]
    fn test_basic() {
        use crate::graph::graph_adjacency_list::Graph;
        let mut graph: Graph<(), ()> = Graph::new();
        let a = graph.add_node(()); // node with no weight
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        let f = graph.add_node(());
        let g = graph.add_node(());
        let h = graph.add_node(());
        // z will be in another connected component
        #[allow(unused_variables)]
        let z = graph.add_node(());

        for edge in [
            (a, b),
            (b, c),
            (c, d),
            (d, a),
            (e, f),
            (b, e),
            (f, g),
            (g, h),
            (h, e),
        ] {
            graph.add_edge(edge.0, edge.1, ());
        }

        let expected_res: HashMap<usize, usize> = [
            (a, 3),
            (b, 0),
            (c, 1),
            (d, 2),
            (e, 1),
            (f, 2),
            (g, 3),
            (h, 4),
        ]
        .iter()
        .cloned()
        .collect();
        let res = dijkstra(&graph, b, None, |_| 1);
        assert_eq!(res, expected_res);
    }
}
