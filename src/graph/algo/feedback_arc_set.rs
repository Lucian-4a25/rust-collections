// Feedback arc set (or Feedback edge set, abbr. Fas) is a set of edges when
// they are removed from graph, could make the whole graph acyclic.

// Solution: We could use a depth-first-search method to iterate every node in the path,
// this is a different kind of depth-first-search method, which won't skip visited node,
// it will only end until all nodes in the queue is searched. And this special depth-first-search
// will push every level's nodes into the Vec, and record all the nodes along the path,
// if a path is over, we will remove the last node from the record.

// If we find a node which appeared in the queue, then it indicated from that position in the Vec,
// there is a cycle to the end of the queue. We could record this cycle ring's path.
// and skip this node from depth-first search, or if we don't skip from this point,
// our depth-first search will never be able to end. And because we do depth-first search
// from that node, and we will go back to that node again to iterate its others children nodes,
// so it's okay for us to skip this duplicate node.

// Optimize: when we already find a cycle in the graph, if next time we enter a mid-point
// in the path, and if the current path has another same point with that path, we could assert
// that we already find a cycle in the graph, and record it.
use crate::graph::visit::{EdgeRef, IntoEdgeReferences};
use std::fmt::Debug;
use std::{
    collections::{HashMap, LinkedList, VecDeque},
    hash::Hash,
    vec,
};

#[allow(dead_code)]
/// find feedback arc set which are removed to make graph acyclic using greedy algorithm
pub fn greedy_feedback_arc_set<G>(g: G) -> Vec<G::EdgeId>
where
    G: IntoEdgeReferences,
    G::NodeId: Copy + Hash + Eq + Debug,
{
    let (mut s1, mut s2) = (VecDeque::new(), VecDeque::new());
    let edges_iter = g.edge_references();
    let mut node_map = HashMap::new();
    for edge_ref in edges_iter {
        let source = edge_ref.source();
        let target = edge_ref.target();
        node_map
            .entry(source)
            .or_insert_with(|| Node::new(source))
            .out_edges
            .push(target);

        node_map
            .entry(target)
            .or_insert_with(|| Node::new(target))
            .in_edges
            .push(source);
    }

    // init the edge num of nodes, and put them into different buckets
    let mut node_container = NodeContainer {
        node_map: HashMap::new(),
        buckets: Buckets {
            sinks: vec![],
            sources: vec![],
            positive_buckets: vec![],
            negative_buckets: vec![],
        },
    };
    node_container.init_buckets(node_map);

    loop {
        let mut updated = false;
        while let Some(n) = node_container.pop_sink_node() {
            s2.push_front(n);
            updated = true;
            println!("pop sink node: {:?}", n);
        }

        while let Some(n) = node_container.pop_source_node() {
            s1.push_back(n);
            updated = true;
            println!("pop source node: {:?}", n);
        }

        // choose a max value of (outdegree - indegree), we couldn't iterate whole map to find
        // the target node everytime, that's very low inefficient.
        if let Some(removal) = node_container.pop_max_degree_node() {
            s1.push_back(removal);
            updated = true;
            println!("pop max degree node: {:?}", removal);
        }

        if !updated {
            break;
        }
    }

    s1.extend(s2);
    println!("sequences: {:?}", s1.iter());
    let node_num = s1.len();
    let sequences: HashMap<_, _> = s1.into_iter().zip(0..node_num).collect();
    // retrive edges according to the sequences
    let mut edges = vec![];
    for e in g.edge_references() {
        let target = e.target();
        let source = e.source();
        if sequences[&source] >= sequences[&target] {
            edges.push(e.id());
        }
    }

    edges
}

/// single node represention in the GR algorithm
#[derive(Default, Debug)]
struct Node<I> {
    node_id: I,
    out_edges_num: usize,
    in_edges_num: usize,
    out_edges: Vec<I>,
    in_edges: Vec<I>,
}

impl<I> Node<I> {
    fn new(node_id: I) -> Self {
        Self {
            node_id,
            out_edges_num: 0,
            in_edges_num: 0,
            out_edges: vec![],
            in_edges: vec![],
        }
    }
}

#[derive(Default, Debug)]
struct NodeContainer<I: Debug> {
    node_map: HashMap<I, Node<I>>,
    buckets: Buckets<I>,
}

#[derive(Debug, Default)]
struct Buckets<I> {
    sinks: Vec<I>,
    sources: Vec<I>,
    positive_buckets: Vec<LinkedList<I>>,
    negative_buckets: Vec<LinkedList<I>>,
}

impl<I: Copy + Eq + Hash + Debug> Buckets<I> {
    fn pop_sink(&mut self) -> Option<I> {
        self.sinks.pop()
    }

    fn pop_source(&mut self) -> Option<I> {
        self.sources.pop()
    }

    fn pop_max_degree_node(&mut self) -> Option<I> {
        if let Some(list) = self
            .positive_buckets
            .iter_mut()
            .rev()
            .chain(self.negative_buckets.iter_mut())
            .find(|linked| !linked.is_empty())
        {
            let first_node = list.pop_front().unwrap();
            Some(first_node)
        } else {
            None
        }
    }

    fn add_sink(&mut self, i: I) {
        println!("add sink node: {:?}", i);
        self.sinks.push(i);
    }

    fn add_source(&mut self, i: I) {
        println!("add source node: {:?}", i);
        self.sources.push(i);
    }

    fn add_positive(&mut self, bucket: usize, i: I) {
        println!("add positive bucket {}, i: {:?}", bucket, i);
        if bucket >= self.positive_buckets.len() {
            self.positive_buckets
                .resize_with(bucket + 1, || LinkedList::new());
        }
        self.positive_buckets[bucket].push_front(i);
    }

    fn remove_positive(&mut self, bucket: usize, i: I) {
        println!("remove positive bucket {}, i: {:?}", bucket, i);
        let bucket = &mut self.positive_buckets[bucket];
        let pos = bucket.iter().position(|n| n == &i).unwrap();
        bucket.remove(pos);
    }

    fn add_negative(&mut self, bucket: usize, i: I) {
        println!("add negative bucket {}, i: {:?}", bucket, i);
        if bucket >= self.negative_buckets.len() {
            self.negative_buckets
                .resize_with(bucket + 1, || LinkedList::new());
        }
        self.negative_buckets[bucket].push_front(i);
    }

    fn remove_negative(&mut self, bucket: usize, i: I) {
        println!("remove negative bucket {}, i: {:?}", bucket, i);
        let bucket = &mut self.negative_buckets[bucket];
        let pos = bucket.iter().position(|n| n == &i).unwrap();
        bucket.remove(pos);
    }

    /// remove node from the buckets if nodes is not located in sinks or sources
    fn remove_node(&mut self, node: &Node<I>) {
        let (in_edges_num, out_edges_num) = (node.in_edges_num, node.out_edges_num);
        let (mut delta, overflow) = out_edges_num.overflowing_sub(in_edges_num);
        if overflow {
            delta = usize::MAX - delta;
        }
        if overflow {
            self.remove_negative(delta, node.node_id);
        } else {
            self.remove_positive(delta, node.node_id);
        }
    }

    fn add_node(&mut self, node: &Node<I>) {
        let in_edges_num = node.in_edges_num;
        let out_edges_num = node.out_edges_num;
        if in_edges_num == 0 {
            self.add_source(node.node_id);
        } else if out_edges_num == 0 {
            self.add_sink(node.node_id);
        } else {
            let (mut delta, overflow) = out_edges_num.overflowing_sub(in_edges_num);
            if overflow {
                delta = usize::MAX - delta
            };
            if overflow {
                self.add_negative(delta, node.node_id);
            } else {
                self.add_positive(delta, node.node_id);
            }
        }
    }
}

/// a node container where we could retrive node from it.
impl<I: Copy + Eq + Hash + Debug> NodeContainer<I> {
    fn init_buckets(&mut self, mut node_map: HashMap<I, Node<I>>) {
        for (_, v) in node_map.iter_mut() {
            v.in_edges_num = v.in_edges.len();
            v.out_edges_num = v.out_edges.len();
            self.add_node(v);
        }
        self.node_map = node_map;
    }

    fn add_node(&mut self, node: &Node<I>) {
        self.buckets.add_node(node);
    }

    #[inline]
    fn update_removal_neighbors(&mut self, node: Node<I>) {
        let node_id = node.node_id;
        // remove node's related edges
        for from_node in node.in_edges {
            // self loop
            if from_node == node_id {
                continue;
            }

            // only decrease the num, when current node is removed, if related node is
            // removed, it will be removed in the node_map, we can just ignore it.
            self.node_map.entry(from_node).and_modify(|node_mut| {
                if node_mut.in_edges_num == 0 || node_mut.out_edges_num == 0 {
                    node_mut.out_edges_num -= 1;
                    return;
                }
                self.buckets.remove_node(node_mut);
                node_mut.out_edges_num -= 1;
                self.buckets.add_node(node_mut);
            });
        }

        for to_node in node.out_edges {
            if to_node == node_id {
                continue;
            }

            self.node_map.entry(to_node).and_modify(|n| {
                if n.in_edges_num == 0 || n.out_edges_num == 0 {
                    n.in_edges_num -= 1;
                    return;
                }
                self.buckets.remove_node(n);
                n.in_edges_num -= 1;
                self.buckets.add_node(n);
            });
        }
    }

    fn pop_sink_node(&mut self) -> Option<I> {
        if let Some(s) = self.buckets.pop_sink() {
            let removal = self
                .node_map
                .remove(&s)
                .expect("the removed node must exist in the container");
            self.update_removal_neighbors(removal);

            Some(s)
        } else {
            None
        }
    }

    fn pop_source_node(&mut self) -> Option<I> {
        if let Some(s) = self.buckets.pop_source() {
            let removal = self
                .node_map
                .remove(&s)
                .expect("the removed node must exist in the container");
            self.update_removal_neighbors(removal);

            Some(s)
        } else {
            None
        }
    }

    fn pop_max_degree_node(&mut self) -> Option<I> {
        if let Some(n) = self.buckets.pop_max_degree_node() {
            let removal = self
                .node_map
                .remove(&n)
                .expect("the removed node must exist in the container");
            self.update_removal_neighbors(removal);

            Some(n)
        } else {
            None
        }
    }
}
