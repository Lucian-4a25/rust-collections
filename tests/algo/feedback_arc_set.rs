use rust_collections::graph::algo::feedback_arc_set::greedy_feedback_arc_set;
use rust_collections::graph::algo::is_graph_cyclic;
use rust_collections::graph::graph_adjacency_list::Graph;

/// adopt test example from petgraph
#[test]
fn test_greedy_feedback_arc_set() {
    let mut g = Graph::new();
    for _ in 0..6 {
        g.add_node(());
    }

    //                       x
    //      0 -> 1 -> 2 -> 3 -> 4 -> 5
    //      4 -> 1
    //      5 -> 0
    //      1 -> 3
    for edge in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (4, 1),
        (1, 3),
    ] {
        g.add_edge(edge.0, edge.1, ());
    }

    // check if graph is cyclic before
    assert_eq!(is_graph_cyclic(&g), true);

    let edges = greedy_feedback_arc_set(&g);
    for edge in edges {
        // println!("removed edge: {:?}", edge);
        g.remove_edge(edge);
    }

    // check if graph is cyclic after apply the fas algorithm
    assert_eq!(is_graph_cyclic(&g), false);
}
