use rust_collections::graph::algo::floyd_warshall::floyd_warshall;
use rust_collections::graph::graph_adjacency_list::Graph;
use std::collections::HashMap;

#[test]
/// adopt pet_graph's test example
fn test_basic() {
    let mut graph: Graph<(), (), _> = Graph::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());

    for edge in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] {
        graph.add_edge(edge.0, edge.1, ());
    }

    let weight_map: HashMap<(_, _), i32> = [
        ((a, a), 0),
        ((a, b), 1),
        ((a, c), 4),
        ((a, d), 10),
        ((b, b), 0),
        ((b, c), 2),
        ((b, d), 2),
        ((c, c), 0),
        ((c, d), 2),
    ]
    .iter()
    .cloned()
    .collect();
    //     ----- b --------
    //    |      ^         | 2
    //    |    1 |    4    v
    //  2 |      a ------> c
    //    |   10 |         | 2
    //    |      v         v
    //     --->  d <-------

    let inf = std::i32::MAX;
    let expected_res: HashMap<(_, _), i32> = [
        ((a, a), 0),
        ((a, b), 1),
        ((a, c), 3),
        ((a, d), 3),
        ((b, a), inf),
        ((b, b), 0),
        ((b, c), 2),
        ((b, d), 2),
        ((c, a), inf),
        ((c, b), inf),
        ((c, c), 0),
        ((c, d), 2),
        ((d, a), inf),
        ((d, b), inf),
        ((d, c), inf),
        ((d, d), 0),
    ]
    .iter()
    .cloned()
    .collect();

    let res = floyd_warshall(&graph, |edge| {
        if let Some(weight) = weight_map.get(&(edge.source(), edge.target())) {
            *weight
        } else {
            inf
        }
    })
    .unwrap();

    let nodes = [a, b, c, d];
    for node1 in &nodes {
        for node2 in &nodes {
            assert_eq!(
                res.get(&(*node1, *node2)).unwrap(),
                expected_res.get(&(*node1, *node2)).unwrap()
            );
        }
    }
}
