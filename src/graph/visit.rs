use fixedbitset::FixedBitSet;
use std::{
    collections::{HashSet, VecDeque},
    fmt::Debug,
    hash::Hash,
};

use super::Direction;

pub trait GraphBase {
    type NodeId: Copy;
    type EdgeId: Copy;
}

pub trait GraphRef: Copy + GraphBase {}

pub trait IntoNodeIdentifiers: GraphRef {
    type NodeIdentifiers: Iterator<Item = Self::NodeId>;

    fn node_identifiers(self) -> Self::NodeIdentifiers;
}

pub trait IntoNeiborghbors: GraphRef {
    type Neighbors: Iterator<Item = Self::NodeId>;

    fn neighbors(self, n: Self::NodeId) -> Self::Neighbors;
}

pub trait IntoNeighborsDirected: IntoNeiborghbors {
    type NeighborsDirected: Iterator<Item = Self::NodeId>;

    fn neighbors_directed(self, n: Self::NodeId, d: Direction) -> Self::NeighborsDirected;
}

/// Definition of a Edge ref
pub trait EdgeRef: Copy {
    type NodeId;
    type EdgeId;
    type EdgeWeight;

    // Required methods
    fn source(&self) -> Self::NodeId;
    fn target(&self) -> Self::NodeId;
    fn weight(&self) -> &Self::EdgeWeight;
    fn id(&self) -> Self::EdgeId;
}

// pub trait Some<T> {
//     type N;
// }

pub trait IntoEdgeReferences: GraphRef {
    // example to specify generice param and associated param together
    // type S: Some<Self::NodeId, N = Self::NodeId>;
    type EdgeWeight;
    type EdgeRef: EdgeRef<
        NodeId = Self::NodeId,
        EdgeId = Self::EdgeId,
        EdgeWeight = Self::EdgeWeight,
    >;
    type EdgeReferences: Iterator<Item = Self::EdgeRef>;

    fn edge_references(self) -> Self::EdgeReferences;
}

pub trait IntoEdges: IntoEdgeReferences {
    type Edges: Iterator<Item = Self::EdgeRef>;

    fn edges(self, n: Self::NodeId) -> Self::Edges;
}

pub trait VisitMap<N> {
    fn visit(&mut self, n: N) -> bool;

    fn is_visit(&self, n: N) -> bool;
}

/// use `FixedBitSet` to record the visited node, save much memory space
impl VisitMap<usize> for FixedBitSet {
    fn is_visit(&self, n: usize) -> bool {
        self.contains(n)
    }

    fn visit(&mut self, n: usize) -> bool {
        !self.put(n)
    }
}

/// use 'HashSet' to record the visited node, this is useful for Dok implemention of graph
impl<N> VisitMap<N> for HashSet<N>
where
    N: Hash + Eq,
{
    fn is_visit(&self, n: N) -> bool {
        self.contains(&n)
    }

    fn visit(&mut self, n: N) -> bool {
        self.insert(n)
    }
}

pub trait Visitable: GraphBase {
    type Map: VisitMap<Self::NodeId>;

    fn visit_map(&self) -> Self::Map;

    fn reset_map(&self, map: &mut Self::Map);
}

pub trait NodeCount {
    fn node_count(self) -> usize;
}

/// convert graph's NodeId to numberic index
pub trait NodeIndexable: GraphBase {
    fn node_bound(&self) -> usize;

    fn to_index(&self, n: Self::NodeId) -> usize;

    fn from_index(&self, i: usize) -> Self::NodeId;
}

pub trait GraphProp: GraphBase {
    fn is_directed(self) -> bool;
}

/// second general version of deep first search
pub struct Dfs<N, VM> {
    pub queue: VecDeque<N>,
    pub last: Option<N>,
    pub discovered: VM,
}

#[allow(dead_code)]
impl<N, VM> Dfs<N, VM>
where
    VM: VisitMap<N>,
    N: Copy,
{
    pub fn new<G>(graph: G, start: N) -> Self
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        let mut queue = VecDeque::new();
        queue.push_back(start);

        Self {
            discovered: graph.visit_map(),
            last: None,
            queue,
        }
    }

    pub fn from_parts(queue: VecDeque<N>, discovered: VM) -> Self {
        Self {
            queue,
            discovered,
            last: None,
        }
    }

    /// clear the visit state
    pub fn reset<G>(&mut self, graph: G)
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        self.queue.clear();
        graph.reset_map(&mut self.discovered);
    }

    /// create a new dfs instance using the graph's visitor map, and no stack
    pub fn empty<G>(graph: G) -> Self
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        Self {
            queue: VecDeque::new(),
            discovered: graph.visit_map(),
            last: None,
        }
    }

    /// keep the visitor state, clear the visitor stack, and restart the dfs from a particular node
    pub fn move_to<G>(&mut self, n: N) {
        self.queue.clear();
        self.queue.push_back(n);
        self.last = None;
    }

    /// return the next node in the dfs process
    pub fn next<G>(&mut self, graph: G) -> Option<N>
    where
        G: IntoNeiborghbors<NodeId = N>,
    {
        // retrive the lastest child nodes of last until we call `next`
        if let Some(last) = self.last.take() {
            let into_neighbors = IntoNeiborghbors::neighbors(graph, last);
            for n in into_neighbors {
                if !self.discovered.is_visit(n) {
                    self.queue.push_front(n);
                }
            }
        }

        // find the next node
        while let Some(next_node) = self.queue.pop_front() {
            if self.discovered.visit(next_node) {
                self.last = Some(next_node);
                return Some(next_node);
            }
        }

        None
    }
}

pub struct DfsPostOrder<N, VM> {
    queue: Vec<N>,
    discovered: VM,
    founded: VM,
}

#[allow(dead_code)]
impl<N, VM> DfsPostOrder<N, VM>
where
    VM: VisitMap<N>,
    N: Copy + Debug,
{
    pub fn new<G>(graph: G, start: N) -> Self
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        let mut queue = Vec::new();
        queue.push(start);
        Self {
            discovered: graph.visit_map(),
            founded: graph.visit_map(),
            queue,
        }
    }

    /// clear the visit state
    pub fn reset<G>(&mut self, graph: G)
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        self.queue.clear();
        graph.reset_map(&mut self.founded);
        graph.reset_map(&mut self.discovered);
    }

    /// create a new dfs instance using the graph's visitor map, and no stack
    pub fn empty<G>(graph: G) -> Self
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        Self {
            queue: Vec::new(),
            discovered: graph.visit_map(),
            founded: graph.visit_map(),
        }
    }

    /// keep the discover and found state, clear the visitor stack, and restart the dfs from a particular node
    pub fn move_to<G>(&mut self, n: N) {
        self.queue.clear();
        self.queue.push(n);
    }

    /// return the next node in the dfs process
    pub fn next<G>(&mut self, graph: G) -> Option<N>
    where
        G: IntoNeiborghbors<NodeId = N>,
    {
        while let Some(&search_node) = self.queue.last() {
            // println!("search node: {:?}", search_node);
            if self.discovered.visit(search_node) {
                for n in graph.neighbors(search_node) {
                    if !self.discovered.is_visit(n) {
                        self.queue.push(n);
                    }
                }
            } else {
                self.queue.pop();
                // There are servral reasons to use found:
                // 1. when move to a new position manully, we could use found to avoid
                // to iterate same node before
                // 2. there may be more than one node to connect same node, and that node is not visited,
                // so for that node, it's possible to be added twice in the queue
                if self.founded.visit(search_node) {
                    return Some(search_node);
                }
            }
        }

        None
    }
}

pub struct Bfs<N, VM> {
    last: Option<N>,
    queue: VecDeque<N>,
    discovered: VM,
}

#[allow(dead_code)]
impl<N, VM> Bfs<N, VM>
where
    VM: VisitMap<N>,
    N: Copy,
{
    pub fn new<G>(graph: G, start: N) -> Self
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        let mut queue = VecDeque::new();
        queue.push_back(start);
        Self {
            last: None,
            discovered: graph.visit_map(),
            queue,
        }
    }

    pub fn from_parts(queue: VecDeque<N>, discovered: VM) -> Self {
        Self {
            queue,
            discovered,
            last: None,
        }
    }

    /// clear the visit state
    pub fn reset<G>(&mut self, graph: G)
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        self.queue.clear();
        self.last = None;
        graph.reset_map(&mut self.discovered);
    }

    /// create a new dfs instance using the graph's visitor map, and no stack
    pub fn empty<G>(graph: G) -> Self
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        Self {
            last: None,
            queue: VecDeque::new(),
            discovered: graph.visit_map(),
        }
    }

    /// keep the visitor state, clear the visitor stack, and restart the dfs from a particular node
    pub fn move_to<G>(&mut self, n: N) {
        self.queue.clear();
        self.queue.push_back(n);
    }

    /// return the next node in the dfs process
    pub fn next<G>(&mut self, graph: G) -> Option<N>
    where
        G: IntoNeiborghbors<NodeId = N>,
    {
        // retrive the lastest child nodes of last until we call `next`
        if let Some(last) = self.last.take() {
            let into_neighbors = IntoNeiborghbors::neighbors(graph, last);
            for n in into_neighbors {
                if !self.discovered.is_visit(n) {
                    self.queue.push_back(n);
                }
            }
        }

        // find the next node
        while let Some(next_node) = self.queue.pop_front() {
            if self.discovered.visit(next_node) {
                self.last = Some(next_node);
                return Some(next_node);
            }
        }

        None
    }
}

/// a implenmention of topological ordering graph traversal, which guarantee the
/// node which have an edge to another node must be visited before it.
pub struct Topological<N, VM> {
    tovisit: Vec<N>,
    ordered: VM,
}

#[allow(dead_code)]
impl<N, VM> Topological<N, VM>
where
    VM: VisitMap<N>,
    N: Copy,
{
    pub fn new<G>(graph: G) -> Self
    where
        G: Visitable<NodeId = N, Map = VM> + IntoNodeIdentifiers + IntoNeighborsDirected,
    {
        let mut init = Self::empty(graph);
        init.extends_with_idxs(graph);
        init
    }

    fn extends_with_idxs<G>(&mut self, graph: G)
    where
        G: IntoNodeIdentifiers + IntoNeighborsDirected<NodeId = N>,
    {
        self.tovisit
            .extend(graph.node_identifiers().filter(|n: &N| {
                graph
                    .neighbors_directed(*n, Direction::Incoming)
                    .next()
                    .is_none()
            }));
    }

    pub fn empty<G>(graph: G) -> Self
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        Self {
            tovisit: Default::default(),
            ordered: graph.visit_map(),
        }
    }

    pub fn reset<G>(&mut self, graph: G)
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        self.tovisit.clear();
        graph.reset_map(&mut self.ordered);
    }

    /// get node index in topological ordering
    pub fn next<G>(&mut self, graph: G) -> Option<N>
    where
        G: IntoNeighborsDirected<NodeId = N>,
    {
        while let Some(n) = self.tovisit.pop() {
            if self.ordered.visit(n) {
                // the incomming edges must be visited, we only need to care about the outcoming edge
                for i in graph.neighbors_directed(n, Direction::Outcoming) {
                    // put the node which all of its incoming node is visited to the list
                    if graph
                        .neighbors_directed(i, Direction::Incoming)
                        .all(|n| self.ordered.is_visit(n))
                        && !self.ordered.is_visit(i)
                    {
                        self.tovisit.push(i);
                    }
                }

                return Some(n);
            }
        }

        None
    }
}

#[cfg(test)]
mod test_visit_func {
    use super::*;
    use crate::graph::graph_adjacency_list::Graph;

    fn create_basic_graph() -> Graph<&'static str, usize> {
        let mut graph = Graph::new();
        let famous_physicist = [
            "Galileo", "Newton", "Faraday", "Maxwell", "Einstein", "Planck",
        ];
        for name in famous_physicist.clone() {
            graph.add_node(name);
        }

        graph.add_edge(0, 1, 7);
        graph.add_edge(1, 2, 1);
        graph.add_edge(2, 3, 2);
        graph.add_edge(3, 4, 3);
        graph.add_edge(0, 3, 4);
        graph.add_edge(1, 4, 5);
        graph.add_edge(1, 5, 6);

        graph
    }

    #[test]
    fn test_basic_dfs() {
        let graph = create_basic_graph();
        let mut dfs = Dfs::new(&graph, 0);
        let mut counter = 0;
        while let Some(n) = dfs.next(&graph) {
            let expected_n = match counter {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => 3,
                4 => 4,
                5 => 5,
                _ => unreachable!(),
            };
            assert_eq!(expected_n, n);
            counter += 1;
        }
        assert_eq!(counter, 6);
    }

    #[test]
    fn test_basic_dfs_postorder() {
        let graph = create_basic_graph();
        let mut dfs = DfsPostOrder::new(&graph, 0);
        let mut counter = 0;
        while let Some(n) = dfs.next(&graph) {
            let expected_n = match counter {
                0 => 4,
                1 => 3,
                2 => 2,
                3 => 5,
                4 => 1,
                5 => 0,
                _ => unreachable!(),
            };
            assert_eq!(expected_n, n);
            counter += 1;
        }
        assert_eq!(counter, 6);
    }

    #[test]
    fn test_basic_bfs() {
        let graph = create_basic_graph();
        let mut dfs = Bfs::new(&graph, 0);
        let mut counter = 0;
        while let Some(n) = dfs.next(&graph) {
            let expected_n = match counter {
                0 => 0,
                1 => 3,
                2 => 1,
                3 => 4,
                4 => 5,
                5 => 2,
                _ => unreachable!(),
            };
            assert_eq!(expected_n, n);
            counter += 1;
        }
        assert_eq!(counter, 6);
    }

    #[test]
    fn test_basic_topological() {
        let graph = create_basic_graph();
        let mut topo = Topological::new(&graph);

        let mut counter = 0;
        while let Some(n) = topo.next(&graph) {
            let expected_n = match counter {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => 3,
                4 => 4,
                5 => 5,
                _ => unreachable!(),
            };
            assert_eq!(expected_n, n);
            counter += 1;
        }
        assert_eq!(counter, 6);
    }

    /// direction: from top to bottom
    ///        A     B
    ///      /  \  /  \
    ///    C     D     E
    ///      \  /     /
    ///       F     G
    ///         \  /
    ///          H
    #[test]
    fn test_complex_topologic() {
        let mut graph = Graph::new();
        let names = ["A", "B", "C", "D", "E", "F", "G", "H"];
        for name in names {
            graph.add_node(name);
        }

        graph.add_edge(0, 2, ());
        graph.add_edge(0, 3, ());
        graph.add_edge(1, 3, ());
        graph.add_edge(1, 4, ());
        graph.add_edge(2, 5, ());
        graph.add_edge(3, 5, ());
        graph.add_edge(4, 6, ());
        graph.add_edge(5, 7, ());
        graph.add_edge(6, 7, ());

        let mut topo = Topological::new(&graph);

        let mut counter = 0;
        // it's B -> E -> G -> A -> C -> D -> F -> H
        while let Some(n) = topo.next(&graph) {
            let expected_n = match counter {
                0 => 1,
                1 => 4,
                2 => 6,
                3 => 0,
                4 => 2,
                5 => 3,
                6 => 5,
                7 => 7,
                _ => unreachable!(),
            };
            assert_eq!(expected_n, n);
            counter += 1;
        }
        assert_eq!(counter, 8);
    }
}
