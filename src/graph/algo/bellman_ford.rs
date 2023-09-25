use super::WeightMeasure;
use crate::graph::visit::{
    EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable, VisitMap, Visitable,
};
use std::hash::Hash;

#[allow(dead_code)]
pub struct Paths<D, P> {
    distances: Vec<D>,
    predecessors: Vec<Option<P>>,
}

#[derive(Debug)]
pub struct NegativeCycleErr;

#[allow(dead_code)]
/// bellman ford shortest path algorithm, could be used to find shortest path in a negative weight graph.
/// A little like normal mind to do things, it iterate all possible edges every time, to check if there is
/// a short way from source to a reachable node. And use a loop to iterate until all node is visited, to
/// make sure all possible shortest path is considered, the max needed loop times is node_count - 1.
/// Although this algorithm solves the problem of caclulate shortest path in negative weight edge graph,
/// but it  has a very bad performance in big graph, because it use a O(|V| * |E|) time complexity.
///
/// Analysis:
/// 1. There should be improvement space for this algorithm, because it iterate all edges
/// in every single loop, but actully that's not neccessary, we could only iterate all
/// edges of all visited nodes every time.
/// 2. Another way we can consider to improve the performance is to avoid unneccsarry iterate
/// if we don't see negative edge weight in last loop.
pub fn bellman_ford<G>(
    graph: G,
    source: G::NodeId,
) -> Result<Paths<G::EdgeWeight, G::NodeId>, NegativeCycleErr>
where
    G: Visitable + IntoNodeIdentifiers + NodeCount + NodeIndexable + IntoEdges,
    G::NodeId: Hash + Eq,
    G::EdgeWeight: WeightMeasure,
{
    let (distances, path_tracker) = construct_bellman(graph, source);

    // check if there is negative cycle in the graph
    for node in graph.node_identifiers() {
        for edge in graph.edges(node) {
            let to_id = graph.to_index(edge.target());
            let from_id = graph.to_index(node);
            if distances[to_id] > *edge.weight() + distances[from_id] {
                return Err(NegativeCycleErr);
            }
        }
    }

    Ok(Paths {
        distances,
        predecessors: path_tracker,
    })
}

/// find a negative cycle from source node
#[allow(dead_code)]
pub fn find_negative_cycle<G>(graph: G, source: G::NodeId) -> Option<Vec<G::NodeId>>
where
    G: Visitable + IntoNodeIdentifiers + NodeCount + NodeIndexable + IntoEdges,
    G::NodeId: Hash + Eq,
    G::EdgeWeight: WeightMeasure,
{
    let (distances, path_tracker) = construct_bellman(graph, source);
    let mut paths = Vec::new();

    'outer: for node in graph.node_identifiers() {
        for edge in graph.edges(node) {
            let to_id = graph.to_index(edge.target());
            let from_id = graph.to_index(node);
            if distances[to_id] > *edge.weight() + distances[from_id] {
                let start_node = node;
                let mut visited = graph.visit_map();
                let mut node_idx = graph.to_index(start_node);

                loop {
                    let last_node = match path_tracker[node_idx] {
                        Some(last_node) => last_node,
                        // FIXME: Does self loop will make us get None?
                        None => graph.from_index(node_idx),
                    };
                    // the start_node is in the negative cycle, we could just return paths.
                    if last_node == start_node {
                        paths.push(start_node);
                        break 'outer;
                    }
                    // case when a node outside the negative cycle is found first, we couldn't know
                    // how many nodes between the start node and first cycle node, so we need to
                    // check from current paths
                    else if visited.is_visit(last_node) {
                        let last_pos = paths.iter().position(|&n| n == last_node).unwrap();
                        paths = paths.split_off(last_pos);
                        break 'outer;
                    }

                    visited.visit(last_node);
                    paths.push(last_node);
                    node_idx = graph.to_index(last_node);
                }
            }
        }
    }

    if !paths.is_empty() {
        paths.reverse();
        return Some(paths);
    }

    None
}

fn construct_bellman<G>(graph: G, source: G::NodeId) -> (Vec<G::EdgeWeight>, Vec<Option<G::NodeId>>)
where
    G: IntoEdges + NodeCount + NodeIndexable + IntoNodeIdentifiers,
    G::NodeId: Hash + Eq,
    G::EdgeWeight: WeightMeasure,
{
    let node_count = graph.node_count();
    let mut distance = vec![G::EdgeWeight::infinite(); node_count];
    let mut path_tracker = vec![None; node_count];
    let source_idx = graph.to_index(source);
    distance[source_idx] = G::EdgeWeight::default();

    for _ in 0..node_count - 1 {
        let mut updated = false;

        for node in graph.node_identifiers() {
            let from_idx = graph.to_index(node);

            for edge in graph.edges(node) {
                let target_node = edge.target();
                let to_id = graph.to_index(target_node);
                let current_distance = distance[from_idx] + *edge.weight();
                if distance[to_id] > current_distance {
                    distance[to_id] = current_distance;
                    path_tracker[to_id] = Some(node);
                    updated = true;
                }
            }
        }

        if !updated {
            break;
        }
    }

    (distance, path_tracker)
}

#[cfg(test)]
mod test_bellman_fords {
    use super::*;

    /// copy the test suit from pet_graph
    #[test]
    fn test_find_negative_cycle() {
        use crate::graph::graph_adjacency_list::Graph;
        let mut g: Graph<usize, f64> = Graph::new();
        for i in 0..5 {
            g.add_node(i);
        }
        for edge in [
            (0, 1, 1.),
            (0, 2, 1.),
            (0, 3, 1.),
            (1, 3, 1.),
            (2, 1, 1.),
            (3, 2, -3.),
        ] {
            g.add_edge(edge.0, edge.1, edge.2);
        }

        let path = find_negative_cycle(&g, 0);
        assert_eq!(path, Some([1, 3, 2].to_vec()));
    }

    /// copy the test suit from pet_graph
    #[test]
    #[allow(unused_variables)]
    fn test_bellman_fords() {
        use crate::graph::graph_adjacency_list::Graph;

        let mut g = Graph::new();
        let a = g.add_node(()); // node with no weight
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        let e = g.add_node(());
        let f = g.add_node(());
        for e in [
            (0, 1, 2.0),
            (0, 3, 4.0),
            (1, 2, 1.0),
            (1, 5, 7.0),
            (2, 4, 5.0),
            (4, 5, 1.0),
            (3, 4, 1.0),
        ] {
            g.add_edge(e.0, e.1, e.2);
        }

        // Graph represented with the weight of each edge
        //
        //     2       1
        // a ----- b ----- c
        // | 4     | 7     |
        // d       f       | 5
        // | 1     | 1     |
        // \------ e ------/

        let path = bellman_ford(&g, a);
        assert!(path.is_ok());
        let path = path.unwrap();
        assert_eq!(path.distances, vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            path.predecessors,
            vec![None, Some(a), Some(b), Some(a), Some(d), Some(e)]
        );

        // Node f (indice 5) can be reach from a with a path costing 6.
        // Predecessor of f is Some(e) which predecessor is Some(d) which predecessor is Some(a).
        // Thus the path from a to f is a <-> d <-> e <-> f

        let mut ng = Graph::new_undirected();

        for _ in 0..7 {
            ng.add_node(());
        }

        for e in [
            (0, 1, -2.0),
            (0, 3, -4.0),
            (1, 2, -1.0),
            (1, 5, -25.0),
            (2, 4, -5.0),
            (4, 5, -25.0),
            (3, 4, -1.0),
        ] {
            ng.add_edge(e.0, e.1, e.2);
        }
        assert!(bellman_ford(&ng, 0).is_err());
    }

    #[test]
    #[allow(dead_code, unused_variables)]
    fn test_bellman_fords_with_integer() {
        use crate::graph::graph_adjacency_list::Graph;

        let mut g = Graph::new();
        let a = g.add_node(()); // node with no weight
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        let e = g.add_node(());
        let f = g.add_node(());
        for e in [
            (0, 1, 2),
            (0, 3, 4),
            (1, 2, 1),
            (1, 5, 7),
            (2, 4, 5),
            (4, 5, 1),
            (3, 4, 1),
        ] {
            g.add_edge(e.0, e.1, e.2);
        }

        // Graph represented with the weight of each edge
        //
        //     2       1
        // a ----- b ----- c
        // | 4     | 7     |
        // d       f       | 5
        // | 1     | 1     |
        // \------ e ------/

        let path = bellman_ford(&g, a);
        assert!(path.is_ok());
        let path = path.unwrap();
        assert_eq!(path.distances, vec![0, 2, 3, 4, 5, 6]);
        assert_eq!(
            path.predecessors,
            vec![None, Some(a), Some(b), Some(a), Some(d), Some(e)]
        );
    }
}
