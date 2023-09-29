use super::visit::{
    EdgeRef, GetAdjacencyMatrix, GraphBase, GraphProp, GraphRef, IntoEdgeReferences, IntoEdges,
    IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable, Visitable,
};
use super::{Directed, Direction, GraphType, IntoWeightedEdge, UnDirected};
use crate::graph::visit::IntoNeiborghbors;
use fixedbitset::FixedBitSet;
use std::cmp::max as max_num;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range};

/// `N` is the type of Vertex, `E` is the type of Edge,
/// `T` is the Type of Graph, default is directed.
pub struct Graph<N, E, T: GraphType = Directed> {
    nodes: Vec<Node<N>>,
    edges: Vec<Edge<E>>,
    phantomdata: PhantomData<T>,
}

pub struct Edge<E> {
    data: E,
    // start node idx and end node idx
    nodes: [usize; 2],
    /// the first ele is the value of start node's next outcoming edge idx,
    /// the second ele is the value of end node's next incoming edge idx
    next: [usize; 2],
}

impl<E> Edge<E> {
    fn new(start_node: usize, end_node: usize, e: E) -> Self {
        Self {
            data: e,
            nodes: [start_node, end_node],
            next: [usize::MAX; 2],
        }
    }
}

#[derive(Default)]
pub struct Node<N> {
    data: N,
    /// the first ele is the idx value of first outcoming edge,
    /// the second ele is the idx value of first incoming edge
    next: [usize; 2],
}

impl<N> Node<N> {
    fn new(n: N) -> Self {
        Self {
            data: n,
            next: [usize::MAX; 2],
        }
    }
}

#[allow(dead_code)]
impl<N, E> Graph<N, E, Directed> {
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
            phantomdata: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<N, E> Graph<N, E, UnDirected> {
    pub fn new_undirected() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
            phantomdata: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<N, E, T: GraphType> Graph<N, E, T> {
    pub fn with_capacity((n, e): (usize, usize)) -> Self {
        Graph {
            nodes: Vec::with_capacity(n),
            edges: Vec::with_capacity(e),
            phantomdata: PhantomData,
        }
    }

    pub fn from_edges<I>(edges: I) -> Self
    where
        I: Iterator,
        I::Item: IntoWeightedEdge<E, NodeId = usize>,
        N: Default,
    {
        let mut graph = Self::with_capacity((0, 0));
        graph.extends_with_edges(edges);
        graph
    }

    pub fn extends_with_edges<I>(&mut self, i: I)
    where
        I: Iterator,
        I::Item: IntoWeightedEdge<E, NodeId = usize>,
        N: Default,
    {
        let mut iter = i.into_iter();
        let (size_hint, _) = iter.size_hint();
        self.edges.reserve(size_hint);

        while let Some(edge) = iter.next() {
            let (from, to, weight) = edge.into_weighted_edge();
            let max_pos = std::cmp::max(from, to);
            // make sure there is a node for edge
            while max_pos >= self.node_count() {
                self.add_node(N::default());
            }
            self.add_edge(from, to, weight);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }

    pub fn capacity(&self) -> (usize, usize) {
        (self.nodes.len(), self.edges.len())
    }

    pub fn is_directed(&self) -> bool {
        T::is_directed()
    }

    /// Add node into the graph, return the index of new inserted node
    pub fn add_node(&mut self, node: N) -> usize {
        let insertion_pos = self.nodes.len();
        let internal_node = Node::new(node);
        self.nodes.push(internal_node);

        insertion_pos
    }

    /// Add edge into the Graph, return the index of new inserted edge
    pub fn add_edge(&mut self, start_node: usize, end_node: usize, edge: E) -> usize {
        let insertion_pos = self.edges.len();
        // avoid multiple allocation
        if max_num(start_node, end_node) >= self.nodes.len() {
            panic!("unable to find the node to add edge");
        }
        let mut internal_edge = Edge::new(start_node, end_node, edge);
        unsafe {
            let node_mut = self.nodes.as_mut_ptr();
            let (start_node_mut, end_node_mut) =
                (&mut *node_mut.add(start_node), &mut *node_mut.add(end_node));
            internal_edge.next[0] = start_node_mut.next[0];
            internal_edge.next[1] = end_node_mut.next[1];
            start_node_mut.next[0] = insertion_pos;
            end_node_mut.next[1] = insertion_pos;
        }

        self.edges.push(internal_edge);

        insertion_pos
    }

    /// remove one node from graph
    #[allow(dead_code)]
    pub fn remove_node(&mut self, node_idx: usize) -> Option<N> {
        if node_idx >= self.nodes.len() {
            // panic!("invalid removed node position");
            return None;
        }

        let old_node_len = self.nodes.len();
        // removed all related edge of node
        let node_edges = self.nodes.get(node_idx)?.next;
        for (start_edge, d) in node_edges.into_iter().zip([0, 1]) {
            let mut edge_pos = start_edge;
            while let Some(edge) = self.edges.get(edge_pos) {
                let next_edge_pos = edge.next[d];
                // only update the other endpoint's queue info
                self.remove_edge_direction(edge_pos, (1 - d).into());
                edge_pos = next_edge_pos;
            }
        }

        let removed_node = self.nodes.swap_remove(node_idx);
        // update moved node's edge info
        if node_idx != old_node_len - 1 {
            let moved_node = self.nodes.get(node_idx).unwrap();
            for (start_edge, d) in moved_node.next.iter().zip([0, 1]) {
                let mut edge_pos = *start_edge;
                while let Some(edge) = self.edges.get_mut(edge_pos) {
                    edge.nodes[d] = node_idx;
                    edge_pos = edge.next[d];
                }
            }
        }

        Some(removed_node.data)
    }

    /// remove one edge from graph
    pub fn remove_edge(&mut self, edge_idx: usize) -> Option<E> {
        if edge_idx >= self.edges.len() {
            // panic!("invalid edge position");
            println!("invalid removed edge position {}", edge_idx);
            return None;
        }
        let old_len = self.edges.len();
        self.update_node_next_edge(edge_idx, self.edges.get(edge_idx)?.next);

        if edge_idx != old_len - 1 {
            self.update_node_next_edge(old_len - 1, [edge_idx, edge_idx]);
        }

        // reserve most of edge's index
        let removed_edge = self.edges.swap_remove(edge_idx);

        Some(removed_edge.data)
    }

    /// only remove specified direction of one edge, for internal use
    fn remove_edge_direction(&mut self, removed_edge_idx: usize, d: Direction) -> Option<E> {
        let removed_edge = self.edges.get(removed_edge_idx)?;
        let old_edge_len = self.edges.len();
        let d: usize = d.into();
        self.update_node_next_edge_with_direction(
            removed_edge.nodes[d],
            removed_edge_idx,
            removed_edge.next[d],
            d.into(),
        );

        if removed_edge_idx != old_edge_len - 1 {
            self.update_node_next_edge(old_edge_len - 1, [removed_edge_idx, removed_edge_idx]);
        }

        let removed_edge = self.edges.swap_remove(removed_edge_idx);

        Some(removed_edge.data)
    }

    /// remove edge from the linked list
    fn update_node_next_edge(&mut self, removed_edge_idx: usize, new_next_edge_pos: [usize; 2]) {
        let removed_edge = self.edges.get(removed_edge_idx).expect("");
        let edge_nodes = removed_edge.nodes;

        // Q: what if we use a Vec in Node to store the incoming edge and outcoming edge,
        // Will this way be more cache friendly if there are many edges for every node.
        // the drawback of using Vec is we will use more memory to store a Node's Edges.
        // A: No, this way won't be more cache friendly, when we iterate all the edge of a
        // Node, we will know the next position of Edge, if they are inserted sequentily,
        // it's also cache friendly for our CPU.
        // update the related node's next chain
        for (&node_pos, d) in edge_nodes.iter().zip([0, 1]) {
            self.update_node_next_edge_with_direction(
                node_pos,
                removed_edge_idx,
                new_next_edge_pos[d],
                d.into(),
            );
        }
    }

    fn update_node_next_edge_with_direction(
        &mut self,
        node_idx: usize,
        removed_edge_idx: usize,
        new_edge_idx: usize,
        d: Direction,
    ) {
        let d: usize = d.into();
        let node_mut = self.nodes.get_mut(node_idx).unwrap();
        let start_edge_pos = node_mut.next[d];
        if start_edge_pos == removed_edge_idx {
            node_mut.next[d] = new_edge_idx;
            return;
        }
        let mut cur_edge_pos = start_edge_pos;
        loop {
            let e = self.edges.get_mut(cur_edge_pos);
            match e {
                Some(e) => {
                    let next_edge_pos = e.next[d];
                    if next_edge_pos == removed_edge_idx {
                        e.next[d] = new_edge_idx;
                        break;
                    }
                    cur_edge_pos = next_edge_pos;
                }
                None => {
                    panic!("there must be edge pos in the linked list");
                }
            }
        }
    }
}

// define general methods for node and edge
#[allow(dead_code)]
impl<N, E, T: GraphType> Graph<N, E, T> {
    /// find a edge from start node to end node
    pub fn find_edge(&self, start_node_idx: usize, end_node_idx: usize) -> Option<usize> {
        let start_node = self.nodes.get(start_node_idx)?;

        let mut outcoming_edge = start_node.next[0];
        while let Some(edge) = self.edges.get(outcoming_edge) {
            if edge.nodes[1] == end_node_idx {
                return Some(outcoming_edge);
            }
            outcoming_edge = edge.next[0];
        }
        if !self.is_directed() {
            let mut incoming_edge = start_node.next[1];
            while let Some(edge) = self.edges.get(incoming_edge) {
                if edge.nodes[0] == end_node_idx {
                    return Some(incoming_edge);
                }
                incoming_edge = edge.next[1];
            }
        }

        None
    }

    /// check if there is edge between start node and end node
    pub fn contains_edge(&self, start_node_idx: usize, end_node_idx: usize) -> bool {
        self.find_edge(start_node_idx, end_node_idx).is_some()
    }

    // find the source and target node idx
    pub fn edge_endpoints(&self, edge_idx: usize) -> Option<(usize, usize)> {
        let nodes = self.edges.get(edge_idx)?.nodes;
        Some((nodes[0], nodes[1]))
    }

    /// get all index of edge
    pub fn edge_indices(&self) -> EdgeIndices {
        EdgeIndices {
            r: 0..self.edge_count(),
        }
    }

    /// get all references of all edges
    pub fn edge_references(&self) -> EdgeReferences<E> {
        EdgeReferences {
            r: (0..self.edge_count()).collect(),
            edges: &self.edges,
        }
    }

    /// Outcoing Edges for Directed Graph
    /// All Edges for UnDirected Graph
    pub fn edges(&self, n: usize) -> EdgeReferences<E> {
        let mut edges = Vec::new();
        let start_pos = if !self.is_directed() {
            vec![0, 1]
        } else {
            vec![0]
        };

        for i in start_pos {
            let mut next_outcomming = self.nodes.get(n).map(|n| n.next[i]);
            while let Some(edge_pos) = next_outcomming {
                if let Some(edge) = self.edges.get(edge_pos) {
                    edges.push(edge_pos);
                    next_outcomming = Some(edge.next[i]);
                } else {
                    break;
                }
            }
        }

        EdgeReferences {
            r: edges,
            edges: &self.edges,
        }
    }

    pub fn edge_weight(&self, edge_idx: usize) -> Option<&E> {
        self.edges.get(edge_idx).map(|edge| &edge.data)
    }

    pub fn edge_weight_mut(&mut self, edge_idx: usize) -> Option<&mut E> {
        self.edges.get_mut(edge_idx).map(|e| &mut e.data)
    }

    /// iterator all edges to get weigth
    pub fn edge_weights(&self) -> EdgeWeights<'_, E> {
        EdgeWeights {
            edges: self.edges.iter(),
        }
    }

    /// iterator all edges to get weigth
    pub fn edge_weights_mut(&mut self) -> EdgeWeightsMut<'_, E> {
        EdgeWeightsMut {
            edges: self.edges.iter_mut(),
        }
    }

    /// get all of node indices
    pub fn node_indices(&self) -> NodeIndices {
        NodeIndices {
            r: 0..self.node_count(),
        }
    }

    pub fn node_weight(&self, node_idx: usize) -> Option<&N> {
        self.nodes.get(node_idx).map(|node| &node.data)
    }

    pub fn node_weight_mut(&mut self, node_idx: usize) -> Option<&mut N> {
        self.nodes.get_mut(node_idx).map(|node| &mut node.data)
    }

    /// get all of node weights
    pub fn node_weights(&self) -> NodeWeights<'_, N> {
        NodeWeights {
            nodes: self.nodes.iter(),
        }
    }

    /// get all of node weights
    pub fn node_weights_mut(&mut self) -> NodeWeightsMut<'_, N> {
        NodeWeightsMut {
            nodes: self.nodes.iter_mut(),
        }
    }

    pub fn raw_nodes(&self) -> &[Node<N>] {
        &self.nodes
    }

    pub fn raw_edges(&self) -> &[Edge<E>] {
        &self.edges
    }

    pub fn first_edge(&self, node_idx: usize, d: Direction) -> Option<usize> {
        let node = self.nodes.get(node_idx)?;
        let first_edge_idx = node.next[usize::from(d)];
        if first_edge_idx != usize::MAX {
            Some(first_edge_idx)
        } else {
            None
        }
    }

    pub fn next_edge(&self, edge_idx: usize, d: Direction) -> Option<usize> {
        let edge = self.edges.get(edge_idx)?;
        let next_edge_idx = edge.next[usize::from(d)];
        if next_edge_idx != usize::MAX {
            Some(next_edge_idx)
        } else {
            None
        }
    }

    /// get a node's all neighbors, which has a related edge with it.
    pub fn neighbors<'a>(&'a self, node_idx: usize) -> Neighbors<'a, E> {
        let target_node = self.nodes.get(node_idx);
        let (edge_out, edge_in) = if let Some(node) = target_node {
            (
                node.next[0],
                if self.is_directed() {
                    usize::MAX
                } else {
                    node.next[1]
                },
            )
        } else {
            (usize::MAX, usize::MAX)
        };
        Neighbors {
            edges: &self.edges,
            target_node: node_idx,
            edge_out: edge_out,
            edge_in: edge_in,
        }
    }

    /// get a node's all neighbors according to a direction
    pub fn neigobors_directed<'a>(&'a self, node_idx: usize, d: Direction) -> Neighbors<'a, E> {
        let target_node = self.nodes.get(node_idx);
        let (edge_out, edge_in) = if let Some(node) = target_node {
            match d {
                Direction::Incoming => (usize::MAX, node.next[1]),
                Direction::Outcoming => (node.next[0], usize::MAX),
            }
        } else {
            (usize::MAX, usize::MAX)
        };
        Neighbors {
            edges: &self.edges,
            target_node: node_idx,
            edge_out: edge_out,
            edge_in: edge_in,
        }
    }

    /// get a node's all neighbors according to a direction
    pub fn neigobors_undirected<'a>(&'a self, node_idx: usize) -> Neighbors<'a, E> {
        let target_node = self.nodes.get(node_idx);
        let (edge_out, edge_in) = if let Some(node) = target_node {
            (node.next[0], node.next[1])
        } else {
            (usize::MAX, usize::MAX)
        };
        Neighbors {
            edges: &self.edges,
            target_node: node_idx,
            edge_out: edge_out,
            edge_in: edge_in,
        }
    }

    pub fn filter_map<'a, F, G, N2, E2>(&'a mut self, mut f1: F, mut f2: G) -> Graph<N2, E2, T>
    where
        F: FnMut(usize, &'a N) -> Option<N2>,
        G: FnMut(usize, &'a E) -> Option<E2>,
    {
        let mut new_g: _ = Graph::with_capacity((0, 0));
        // map from old node idx to new node idx, usize::MAX mean the old node is removed
        let mut node_idx_map = vec![usize::MAX; self.node_count()];
        self.nodes.iter().enumerate().for_each(|(idx, node)| {
            if let Some(n2) = f1(idx, &node.data) {
                node_idx_map[idx] = new_g.add_node(n2);
            }
        });
        self.edges.iter().enumerate().for_each(|(idx, edge)| {
            let [start_node, end_node] = edge.nodes;
            let (new_start_node, new_end_node) = (node_idx_map[start_node], node_idx_map[end_node]);
            // the node of edge is not removed
            if new_start_node != usize::MAX && new_end_node != usize::MAX {
                if let Some(n2) = f2(idx, &edge.data) {
                    // the node info of edge is saved from new, we don't need to worry about it
                    new_g.add_edge(new_start_node, new_end_node, n2);
                }
            }
        });

        new_g
    }

    /// create a new map, which map all the nodes' data and edges' data
    pub fn map<'a, F, G, N2, E2>(&'a mut self, mut f1: F, mut f2: G) -> Graph<N2, E2, T>
    where
        F: FnMut(usize, &'a N) -> N2,
        G: FnMut(usize, &'a E) -> E2,
    {
        let mut new_g: _ = Graph::with_capacity(self.capacity());
        new_g
            .nodes
            .extend(self.nodes.iter().enumerate().map(|(idx, node)| Node {
                data: f1(idx, &node.data),
                next: node.next,
            }));
        new_g
            .edges
            .extend(self.edges.iter().enumerate().map(|(idx, edge)| Edge {
                data: f2(idx, &edge.data),
                next: edge.next,
                nodes: edge.nodes,
            }));

        new_g
    }
}

/// allow to access node's data by `graph[idx]` syntax
impl<N, E, T: GraphType> Index<usize> for Graph<N, E, T> {
    type Output = N;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index].data
    }
}

impl<N, E, T: GraphType> IndexMut<usize> for Graph<N, E, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index].data
    }
}

pub struct Neighbors<'a, E> {
    edges: &'a [Edge<E>],
    target_node: usize,
    edge_out: usize,
    edge_in: usize,
}

impl<'a, E> Iterator for Neighbors<'a, E> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(edge) = self.edges.get(self.edge_out) {
            let neighbor_idx = edge.nodes[1];
            self.edge_out = edge.next[0];
            return Some(neighbor_idx);
        }
        while let Some(edge) = self.edges.get(self.edge_in) {
            let neighbor_idx = edge.nodes[0];
            self.edge_in = edge.next[1];
            // avoid to count self multiple times for selfloop edge
            if neighbor_idx != self.target_node {
                return Some(neighbor_idx);
            }
        }

        None
    }
}

pub struct NodeIndices {
    r: Range<usize>,
}

impl Iterator for NodeIndices {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.r.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.r.size_hint()
    }
}

pub struct NodeWeights<'a, N> {
    nodes: std::slice::Iter<'a, Node<N>>,
}

impl<'a, N> Iterator for NodeWeights<'a, N> {
    type Item = &'a N;

    fn next(&mut self) -> Option<Self::Item> {
        self.nodes.next().map(|node| &node.data)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.nodes.size_hint()
    }
}

pub struct NodeWeightsMut<'a, N> {
    nodes: std::slice::IterMut<'a, Node<N>>,
}

impl<'a, N> Iterator for NodeWeightsMut<'a, N> {
    type Item = &'a mut N;

    fn next(&mut self) -> Option<Self::Item> {
        self.nodes.next().map(|node| &mut node.data)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.nodes.size_hint()
    }
}

pub struct EdgeIndices {
    r: Range<usize>,
}

impl Iterator for EdgeIndices {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.r.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.r.size_hint()
    }
}

pub struct EdgeReferences<'a, E: 'a> {
    r: Vec<usize>,
    edges: &'a [Edge<E>],
}

impl<'a, E: 'a> Iterator for EdgeReferences<'a, E> {
    type Item = EdgeReference<'a, E>;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.r.pop()?;
        let edge = &self.edges[i];
        Some(EdgeReference {
            nodes: edge.nodes,
            weight: &edge.data,
            idx: i,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.r.len(), None)
    }
}

pub struct EdgeReference<'a, E: 'a> {
    nodes: [usize; 2],
    weight: &'a E,
    idx: usize,
}

impl<'a, E: 'a> Clone for EdgeReference<'a, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, E: 'a> Copy for EdgeReference<'a, E> {}

impl<'a, E: 'a> EdgeRef for EdgeReference<'a, E> {
    type EdgeId = usize;
    type EdgeWeight = E;
    type NodeId = usize;

    fn id(&self) -> Self::EdgeId {
        EdgeReference::index(self)
    }

    fn source(&self) -> Self::NodeId {
        EdgeReference::source(self)
    }

    fn target(&self) -> Self::NodeId {
        EdgeReference::target(self)
    }

    fn weight(&self) -> &Self::EdgeWeight {
        EdgeReference::weight(self)
    }
}

#[allow(dead_code)]
impl<'a, E: 'a> EdgeReference<'a, E> {
    pub fn source(&self) -> usize {
        self.nodes[0]
    }

    pub fn target(&self) -> usize {
        self.nodes[1]
    }

    pub fn weight(&self) -> &'a E {
        self.weight
    }

    pub fn index(&self) -> usize {
        self.idx
    }
}

pub struct EdgeWeights<'a, E> {
    edges: std::slice::Iter<'a, Edge<E>>,
}

impl<'a, E> Iterator for EdgeWeights<'a, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        self.edges.next().map(|e| &e.data)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.edges.size_hint()
    }
}

pub struct EdgeWeightsMut<'a, E> {
    edges: std::slice::IterMut<'a, Edge<E>>,
}

impl<'a, E> Iterator for EdgeWeightsMut<'a, E> {
    type Item = &'a mut E;

    fn next(&mut self) -> Option<Self::Item> {
        self.edges.next().map(|e| &mut e.data)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.edges.size_hint()
    }
}

impl<N, E, T: GraphType> GraphBase for Graph<N, E, T> {
    type EdgeId = usize;
    type NodeId = usize;
}

impl<'a, G> GraphBase for &'a G
where
    G: GraphBase,
{
    type EdgeId = G::EdgeId;
    type NodeId = G::NodeId;
}

impl<'a, G> GraphRef for &'a G where G: GraphBase {}

impl<'a, N, E, T: GraphType> IntoNeiborghbors for &'a Graph<N, E, T> {
    type Neighbors = Neighbors<'a, E>;

    fn neighbors(self, n: Self::NodeId) -> Self::Neighbors {
        Graph::neighbors(self, n)
    }
}

impl<'a, N, E, T: GraphType> IntoNeighborsDirected for &'a Graph<N, E, T> {
    type NeighborsDirected = Neighbors<'a, E>;

    fn neighbors_directed(self, n: Self::NodeId, d: Direction) -> Self::NeighborsDirected {
        Graph::neigobors_directed(self, n, d)
    }
}

// #[test]
// fn test_trait_definition() {
//     use super::*;

//     struct Foo<T>
//     where
//         T: Default,
//     {
//         a: T,
//     }

//     let a: Foo<u32> = Foo {
//         a: Default::default(),
//     };
//     println!("{}", a.a);

//     fn test_trait_definition<G, M>(graph: &G) -> M
//     where
//         G: Visitable,
//         M: Default,
//         // G::Map: Visitable,
//     {
//         println!("do something");
//         graph.visit_map();

//         M::default()
//     }

//     let mut graph = Graph::new();
//     graph.add_node(1);
//     graph.add_edge(0, 0, ());

//     test_trait_definition::<_, u32>(&graph);
// }

impl<N, E, T: GraphType> Visitable for Graph<N, E, T> {
    type Map = FixedBitSet;

    fn reset_map(&self, map: &mut Self::Map) {
        map.clear();
        map.grow(self.node_count());
    }

    fn visit_map(&self) -> Self::Map {
        FixedBitSet::with_capacity(self.node_count())
    }
}

impl<'a, N, E, T: GraphType> Visitable for &'a Graph<N, E, T> {
    type Map = FixedBitSet;

    fn reset_map(&self, map: &mut Self::Map) {
        map.clear();
        map.grow(self.node_count());
    }

    fn visit_map(&self) -> Self::Map {
        FixedBitSet::with_capacity(self.node_count())
    }
}

impl<'a, N, E, T: GraphType> IntoNodeIdentifiers for &'a Graph<N, E, T> {
    type NodeIdentifiers = NodeIndices;

    fn node_identifiers(self) -> Self::NodeIdentifiers {
        Graph::node_indices(self)
    }
}

impl<'a, N, E, T: GraphType> IntoEdgeReferences for &'a Graph<N, E, T> {
    type EdgeWeight = E;
    type EdgeRef = EdgeReference<'a, E>;
    type EdgeReferences = EdgeReferences<'a, E>;

    fn edge_references(self) -> Self::EdgeReferences {
        Graph::edge_references(self)
    }
}

impl<'a, N, E, T: GraphType> IntoEdges for &'a Graph<N, E, T> {
    type Edges = EdgeReferences<'a, E>;

    fn edges(self, n: Self::NodeId) -> Self::Edges {
        Graph::edges(&self, n)
    }
}

impl<'a, N, E, T: GraphType> NodeCount for &'a Graph<N, E, T> {
    fn node_count(self) -> usize {
        Graph::node_count(self)
    }
}

impl<'a, N, E, T: GraphType> NodeIndexable for &'a Graph<N, E, T> {
    fn node_bound(&self) -> usize {
        self.node_count()
    }

    fn from_index(&self, i: usize) -> Self::NodeId {
        i
    }

    fn to_index(&self, n: Self::NodeId) -> usize {
        n
    }
}

impl<'a, N, E, T: GraphType> GraphProp for &'a Graph<N, E, T> {
    fn is_directed(self) -> bool {
        Graph::is_directed(self)
    }
}

/// This could be replaced a structure similiar with CSR to check adjacent relation of nodes,
/// thus we could save more space to just use a pure array.
impl<'a, N, E, T: GraphType> GetAdjacencyMatrix for &'a Graph<N, E, T> {
    type AdjMatrix = HashMap<Self::NodeId, HashSet<Self::NodeId>>;

    fn adjacency_matrix(self) -> Self::AdjMatrix {
        let mut adjmatrix = HashMap::new();
        for node_id in self.node_identifiers() {
            let neighbor_container = adjmatrix.entry(node_id).or_insert_with(|| HashSet::new());
            for neighbor_id in self.neighbors(node_id) {
                neighbor_container.insert(neighbor_id);
            }
        }

        adjmatrix
    }

    fn is_adjacent(self, matrix: &Self::AdjMatrix, a: Self::NodeId, b: Self::NodeId) -> bool {
        matrix
            .get(&a)
            .and_then(|m| Some(m.contains(&b)))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod graph_api {
    use super::*;

    #[test]
    fn test_basic() {
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

        macro_rules! check_graph_basic_prop {
            ($graph: tt) => {
                let mut weight = 7;
                for _ in 0..graph.edge_count() {
                    assert_eq!(graph.remove_edge(0), Some(weight));
                    weight -= 1;
                }

                // remove all of nodes
                for i in 0..graph.node_count() {
                    let name = if i == 0 {
                        famous_physicist[0]
                    } else {
                        famous_physicist[famous_physicist.len() - i]
                    };
                    assert_eq!(graph.remove_node(0), Some(name));
                }
            };
        }

        check_graph_basic_prop!(graph);
        graph.clear();
        let famous_physicist = [
            "Galileo", "Newton", "Faraday", "Maxwell", "Einstein", "Planck",
        ];
        for name in famous_physicist.clone() {
            graph.add_node(name);
        }
        graph.extends_with_edges(
            [
                (0, 1, 7),
                (1, 2, 1),
                (2, 3, 2),
                (3, 4, 3),
                (0, 3, 4),
                (1, 4, 5),
                (1, 5, 6),
            ]
            .into_iter(),
        );
        check_graph_basic_prop!(graph);
    }

    #[test]
    pub fn test_graph_utility_method() {
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

        // find_edge api
        assert_eq!(graph.find_edge(1, 5), Some(6));

        // contains_edge api
        assert_eq!(graph.contains_edge(1, 2), true);
        assert_eq!(graph.contains_edge(0, 2), false);
        assert_eq!(graph.contains_edge(3, 5), false);

        // edge_endpoints api
        assert_eq!(graph.edge_endpoints(0), Some((0, 1)));
        assert_eq!(graph.edge_endpoints(6), Some((1, 5)));
        assert_eq!(graph.edge_endpoints(7), None);

        // edge_indices
        let mut i = 0;
        let iter = graph.edge_indices();
        assert_eq!(iter.size_hint().0, 7);
        for edge_idx in iter {
            assert_eq!(edge_idx, i);
            i += 1;
        }

        // edge_references
        let iter = graph.edge_references();
        assert_eq!(iter.size_hint().0, 7);
        let mut all_edges: Vec<usize> = (1..8).collect();
        for edge_ref in iter {
            let pos = all_edges
                .iter()
                .position(|e| e == edge_ref.weight())
                .unwrap();
            all_edges.remove(pos);
        }
        assert_eq!(all_edges.len(), 0);

        // edges
        let iter = graph.edges(1);
        assert_eq!(iter.size_hint().0, 3);
        let mut all_edges: Vec<usize> = vec![1, 6, 5];
        for edge_ref in iter {
            let pos = all_edges
                .iter()
                .position(|e| e == edge_ref.weight())
                .unwrap();
            all_edges.remove(pos);
        }
        assert_eq!(all_edges.len(), 0);

        let iter = graph.edge_weights().enumerate();
        assert_eq!(iter.size_hint().0, 7);
        for (offset, weigth) in iter {
            if offset == 0 {
                assert_eq!(weigth, &7);
            } else {
                assert_eq!(weigth, &offset);
            }
        }

        let iter = graph.node_indices();
        assert_eq!(iter.size_hint().0, 6);
        let mut i = 0;
        for edge_idx in iter {
            assert_eq!(edge_idx, i);
            i += 1;
        }

        let mut iter = graph.node_weights();
        assert_eq!(iter.size_hint().0, 6);
        for name in famous_physicist.clone() {
            assert_eq!(Some(&name), iter.next());
        }

        // first_edge & next_edge
        assert_eq!(graph.first_edge(0, Direction::Outcoming), Some(4));
        assert_eq!(
            graph.next_edge(
                graph.first_edge(1, Direction::Outcoming).unwrap(),
                Direction::Outcoming
            ),
            Some(5)
        );

        let iter = graph.neighbors(3);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(neighbor_id, 4);
            counter += 1;
        }
        assert_eq!(counter, 1);

        let iter = graph.neighbors(1).enumerate();
        let mut counter = 0;
        for (offset, neighbor_id) in iter {
            assert_eq!(
                neighbor_id,
                if offset == 0 {
                    5
                } else if offset == 1 {
                    4
                } else {
                    2
                }
            );
            counter += 1;
        }
        assert_eq!(counter, 3);

        // neigobors_undirected
        let iter = graph.neigobors_undirected(3).enumerate();
        let mut counter = 0;
        for (offset, neighbor_id) in iter {
            assert_eq!(
                neighbor_id,
                if offset == 0 {
                    4
                } else if offset == 1 {
                    0
                } else {
                    2
                }
            );
            counter += 1;
        }
        assert_eq!(counter, 3);

        // map
        let new_graph: _ = graph.map(
            |idx, d| format!("{}-{}", *d, idx),
            |idx, d| format!("{}-{}", *d, idx),
        );

        let iter = new_graph.edge_weights().enumerate();
        assert_eq!(iter.size_hint().0, 7);
        for (offset, weight) in iter {
            if offset == 0 {
                assert_eq!(weight, "7-0");
            } else {
                assert_eq!(weight, format!("{}-{}", offset, offset).as_str());
            }
        }
        let mut iter = new_graph.node_weights();
        assert_eq!(iter.size_hint().0, 6);
        for (offset, name) in famous_physicist.clone().iter().enumerate() {
            assert_eq!(
                iter.next().map(|s| s.as_str()),
                Some(format!("{}-{}", *name, offset).as_str())
            );
        }

        // filter map
        let filter_graph: _ = graph.filter_map(
            |idx, data| {
                if idx == 0 || idx == 2 {
                    None
                } else {
                    Some(*data)
                }
            },
            |idx, d| {
                if idx == 3 {
                    None
                } else {
                    Some(*d)
                }
            },
        );
        let iter = filter_graph.edge_weights().enumerate();
        assert_eq!(iter.size_hint().0, 2);
        for (offset, weight) in iter {
            assert_eq!(weight, if offset == 0 { &5 } else { &6 });
        }

        // edge_mut api
        // increase one for every edge
        for edge in graph.edge_weights_mut() {
            *edge += 1;
        }
        let iter = graph.edge_weights().enumerate();
        assert_eq!(iter.size_hint().0, 7);
        for (offset, weight) in iter {
            if offset == 0 {
                assert_eq!(weight, &8);
            } else {
                assert_eq!(weight, &(offset + 1));
            }
        }

        let famous_physicist_2 = [
            "_Galileo_",
            "_Newton_",
            "_Faraday_",
            "_Maxwell_",
            "_Einstein_",
            "_Planck_",
        ];
        // node_mut api
        for (offset, node) in graph.node_weights_mut().enumerate() {
            *node = famous_physicist_2[offset];
        }
        let iter = graph.node_weights().enumerate();
        assert_eq!(iter.size_hint().0, 6);
        for (offset, name) in iter {
            assert_eq!(*name, famous_physicist_2[offset]);
        }

        // node_weight
        assert_eq!(graph.node_weight(0), Some(&"_Galileo_"));
        let weight_mut = graph.node_weight_mut(0).unwrap();
        *weight_mut = "__Galileo__";
        assert_eq!(graph.node_weight(0), Some(&"__Galileo__"));

        // edge_weight
        assert_eq!(graph.edge_weight(0), Some(&8));
        let edge_mut = graph.edge_weight_mut(0).unwrap();
        *edge_mut += 1;
        assert_eq!(graph.edge_weight(0), Some(&9));
    }
}
