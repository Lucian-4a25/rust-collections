use rust_collections::graph::algo::dijkstra::dijkstra;
use rust_collections::graph::graph_adjacency_list::Graph;
use std::collections::HashMap;

// a ----> b ----> e ----> f
// ^       |       ^       |
// |       v       |       v
// d <---- c       h <---- g
#[test]
fn test_basic() {
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
