use rust_collections::graph::algo::astar::astar;
use rust_collections::graph::graph_adjacency_list::Graph;

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
