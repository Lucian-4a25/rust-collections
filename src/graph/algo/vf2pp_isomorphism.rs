use crate::graph::visit::{
    GraphBase, GraphData, GraphDataAccess, IntoNeiborghbors, IntoNeighborsDirected,
    IntoNodeIdentifiers, NodeCount, NodeIndexable,
};
use std::{cmp::Ordering, collections::BinaryHeap};

const DEFAULT_NODE_LABEL: usize = 0usize;

trait NodeLabel<G: GraphBase> {
    fn get_node_label(&mut self, g: G, node_id: G::NodeId) -> usize;
}

struct NoNodeLabel;

impl<G: GraphBase> NodeLabel<G> for NoNodeLabel {
    fn get_node_label(&mut self, _g: G, _id: <G as GraphBase>::NodeId) -> usize {
        DEFAULT_NODE_LABEL
    }
}

impl<G, F> NodeLabel<G> for F
where
    G: GraphDataAccess,
    F: FnMut(G::NodeWeight) -> usize,
{
    fn get_node_label(&mut self, g: G, node_id: <G as GraphBase>::NodeId) -> usize {
        let node_weight = g.node_weight(node_id).unwrap();
        self(node_weight)
    }
}

#[allow(dead_code)]
/// check if g0 is isomorphic with g1
/// `label` function is used to get label from Graph Node
/// node_matcher and edge_matcher are both optional for matching
pub fn vf2pp_isomorphsim_label_matching<G0, G1, F0, F1, NM, EM>(
    g0: G0,
    g1: G1,
    mut label0: F0,
    mut label1: F1,
    node_matcher: Option<NM>,
    edge_matcher: Option<EM>,
) -> bool
where
    G0: IntoNeighborsDirected + IntoNodeIdentifiers + NodeIndexable + NodeCount + GraphDataAccess,
    G1: IntoNeighborsDirected + IntoNodeIdentifiers + NodeIndexable + NodeCount + GraphDataAccess,
    F0: FnMut(G0::NodeWeight) -> usize,
    F1: FnMut(G1::NodeWeight) -> usize,
    NM: FnMut(G0::NodeWeight, G1::NodeWeight) -> bool,
    EM: FnMut(G0::EdgeWeight, G1::EdgeWeight) -> bool,
{
    // First, we need to sort the G1's nodes accordding ordering way,
    // * label rarity of each node, the more rarity, the ordering is more high
    // * the degree of each node, the more degree mean ordering is higher
    // * use a bfs way to decide the ordering of nodes, for every level of nodes,
    // except the root level, more connections mean higher ordering, if the connections num is the same,
    // then the degrees decide the ordering, last is the label rarity
    let (g0_node_labels, g0_max_label) = init_graph_nodes_labels(g0, &mut label0);
    let (g1_node_labels, g1_max_label) = init_graph_nodes_labels(g1, &mut label1);
    let max_label = std::cmp::max(g0_max_label, g1_max_label);

    // let label_tmp_2 = vec![0; max_label + 1];
    // reset temporay label data
    // let mut label_tmp_1 = vec![0; max_label + 1];
    let matching_order = init_matching_ordering(g0, &g0_node_labels, g0_max_label);

    // Second, init RnewRinout nums for every nodes in G1, this will be used to cut labels in third stage.
    let (r_new, r_in_out) = init_r_new_inout(g0, &matching_order, &g0_node_labels, g0_max_label);

    // Third, pick candidate according the ordering we build in step 1, and for root node, we need to check
    // every possible node in g1, but for non-root node, the only possible nodes is the mapping of its sibling
    // node's neighbors in g1. And check fesibility and cut impossible label based in RnewRoutin info.
    let mut mapping = vec![usize::MAX; g0.node_count()];
    let mut depth = 0usize;
    // g1_cons[i] mean the number of neighbors of `i`, i is node_idx value of g1
    // usize::MAX indicate it already exist in the mapping
    let mut g1_node_cons = vec![0usize; g1.node_count()];
    let mut g1_candidate_nodes_iter = vec![];

    isomorphism_match(
        g0,
        g1,
        &matching_order,
        &mut mapping,
        &r_in_out,
        &r_new,
        &mut g1_node_cons,
        &mut g1_candidate_nodes_iter,
        &mut depth,
        false,
    )
    .is_some()
}

/// match according the ordering of g0, this should be a pure function so that it could be reused
fn isomorphism_match<'a, G0, G1>(
    g0: G0,
    g1: G1,
    order: &Vec<usize>,
    mapping: &mut Vec<usize>,
    r_in_out: &Vec<Vec<(usize, usize)>>,
    r_new: &Vec<Vec<(usize, usize)>>,
    g1_node_cons: &mut Vec<usize>,
    g1_candidate_nodes_iter: &mut Vec<Box<dyn Iterator<Item = G1::NodeId> + 'a>>,
    depth: &mut usize,
    subgraph: bool,
) -> Option<Vec<usize>>
where
    G0: IntoNeiborghbors + NodeIndexable,
    G1: IntoNeiborghbors + NodeIndexable + IntoNodeIdentifiers + 'a,
{
    'outer: while *depth != usize::MAX {
        let cur_depth = *depth;
        // a matched resutl found
        if cur_depth == order.len() {
            *depth -= 1;
            return Some(mapping.iter().map(|v| v.clone()).collect());
        }
        let n = order[cur_depth];
        // let mapping_in_g1 = mapping[n];

        // There are two possible cases:
        // 1. we need to find a fresh candidate in this depth
        // 2. we need to find a candidate from `g1_candidates_nodes_iter`, and forward based previous position
        let cur_iter_idx = g1_candidate_nodes_iter.len() - 1;
        if cur_depth > cur_iter_idx {
            debug_assert!(cur_depth == cur_iter_idx + 1);
            // it's a fresh start, we first need to check the neighbor of n
            let mut has_mapping_neighbor = false;
            for neighbor in g0.neighbors(g0.from_index(n)) {
                let neighbor_mapping_in_g1 = mapping[g0.to_index(neighbor)];
                if neighbor_mapping_in_g1 != usize::MAX {
                    has_mapping_neighbor = true;
                    g1_candidate_nodes_iter.push(Box::new(
                        g1.neighbors(g1.from_index(neighbor_mapping_in_g1)),
                    ));
                    break;
                }
            }

            if !has_mapping_neighbor {
                g1_candidate_nodes_iter.push(Box::new(g1.node_identifiers()));
            }
        }

        let cur_g1_iter = g1_candidate_nodes_iter.last_mut().unwrap();
        while let Some(m_id) = cur_g1_iter.next() {
            let m = g1.to_index(m_id);
            if g1_node_cons[m] != usize::MAX && check_feasibility::<G0, G1>(n, m) {
                mark_candidate_pair(n, m, depth, mapping, g1_node_cons, g1);
                continue 'outer;
            }
        }
        // no matching found, go backward
        unmark_candidate_pair(
            depth,
            order,
            mapping,
            g1_candidate_nodes_iter,
            g1_node_cons,
            g1,
        );
    }

    #[inline]
    fn mark_candidate_pair<G>(
        n: usize,
        m: usize,
        depth: &mut usize,
        mapping: &mut Vec<usize>,
        g1_node_cons: &mut Vec<usize>,
        g1: G,
    ) where
        G: IntoNeiborghbors + NodeIndexable,
    {
        *depth += 1;
        mapping[n] = m;
        g1_node_cons[m] = usize::MAX;
        for neighbor in g1.neighbors(g1.from_index(m)) {
            let neighbor_idx = g1.to_index(neighbor);
            if g1_node_cons[neighbor_idx] != usize::MAX {
                g1_node_cons[neighbor_idx] += 1;
            }
        }
    }

    #[inline]
    fn unmark_candidate_pair<'a, G>(
        depth: &mut usize,
        matching_order: &Vec<usize>,
        mapping: &mut Vec<usize>,
        g1_candidate_nodes_iter: &mut Vec<Box<dyn Iterator<Item = G::NodeId> + 'a>>,
        g1_node_cons: &mut Vec<usize>,
        g1: G,
    ) where
        G: IntoNeiborghbors + NodeIndexable,
    {
        if *depth == 0 {
            *depth = usize::MAX;
        } else {
            *depth -= 1;
        }
        let m = mapping[matching_order[*depth]];
        mapping[matching_order[*depth]] = usize::MAX;
        g1_node_cons[m] = 0;
        g1_candidate_nodes_iter.pop();
        for neighbor in g1.neighbors(g1.from_index(m)) {
            let neighbor_id = g1.to_index(neighbor);
            let neighbor_con = g1_node_cons[neighbor_id];
            if neighbor_con != usize::MAX && neighbor_con > 0 {
                g1_node_cons[neighbor_id] -= 1;
            } else if neighbor_con == usize::MAX {
                g1_node_cons[m] += 1;
            }
        }
    }

    None
}

/// check the feasiblity of candidate node pair, also cut labels in this method
fn check_feasibility<G0, G1>(n: usize, m: usize) -> bool
where
    G0: IntoNeiborghbors,
    G1: IntoNeiborghbors,
{
    todo!()
}

#[allow(dead_code)]
/// initilize all graph node labels, node_labels[i] mean the corresponding label of node_idx `i`.
/// And the number of a node label is less indicate the label is more rare
fn init_graph_nodes_labels<G, F>(g: G, label: &mut F) -> (Vec<usize>, usize)
where
    G: IntoNodeIdentifiers + NodeCount + NodeIndexable,
    F: NodeLabel<G>,
{
    let node_count = g.node_count();
    let mut max_label = 0;
    let mut node_lables = vec![DEFAULT_NODE_LABEL; node_count];

    for node in g.node_identifiers() {
        let node_idx = g.to_index(node);
        let node_label = NodeLabel::get_node_label(label, g, node);
        node_lables[node_idx] = node_label;
        if max_label < node_label {
            max_label = node_label;
        }
    }

    (node_lables, max_label)
}

#[allow(dead_code)]
/// initilize the ordering of g0, the `label_tmp[i]` mean the number of nodes with same label `i`
fn init_matching_ordering<G>(g: G, node_labels: &Vec<usize>, max_label: usize) -> Vec<usize>
where
    G: IntoNeiborghbors + IntoNodeIdentifiers + NodeCount + NodeIndexable,
{
    let node_count = g.node_count();
    let mut matching_order = vec![0; node_count];
    let mut added = vec![false; node_count];
    let mut node_con_num = vec![0; node_count];
    let mut node_label_num = vec![0usize; max_label + 1];
    // init every node label's number
    for label in node_labels.iter() {
        node_label_num[*label] += 1;
    }

    // init every node's connection number
    for node in g.node_identifiers() {
        let node_id = g.to_index(node);
        for neighbor in g.neighbors(node) {
            let neighbor_id = g.to_index(neighbor);
            node_con_num[neighbor_id] += 1;
            node_con_num[node_id] += 1;
        }
    }

    let mut order_idx = 0usize;
    // We need to choose all possible root nodes, then do BFS search using the choosen node as root node
    for node in g.node_identifiers() {
        let node_id = g.to_index(node);
        if !added[node_id] {
            let mut min_node_id = node_id;
            // select a root node for bfs search ordering
            for inner_node in g.node_identifiers() {
                let inner_node_id = g.to_index(inner_node);
                if added[inner_node_id] {
                    continue;
                }
                match node_label_num[inner_node_id].cmp(&node_label_num[inner_node_id]) {
                    Ordering::Less => {
                        min_node_id = inner_node_id;
                    }
                    Ordering::Equal => {
                        if node_con_num[inner_node_id] > node_con_num[min_node_id] {
                            min_node_id = inner_node_id;
                        }
                    }
                    Ordering::Greater => {}
                }
            }
            bfs_search_ordering(
                g,
                min_node_id,
                &mut order_idx,
                &mut added,
                &mut matching_order,
                &node_con_num,
                node_labels,
                &mut node_label_num,
            );
        }
    }

    struct LayerNode {
        node_idx: usize,
        bfs_cons_num: usize,
        all_cons_num: usize,
        label_num: usize,
    }

    impl Ord for LayerNode {
        fn cmp(&self, other: &Self) -> Ordering {
            match self.bfs_cons_num.cmp(&other.bfs_cons_num) {
                Ordering::Greater => Ordering::Greater,
                Ordering::Less => Ordering::Less,
                Ordering::Equal => {
                    if self.all_cons_num > other.all_cons_num {
                        return Ordering::Greater;
                    } else if self.all_cons_num < other.all_cons_num {
                        return Ordering::Less;
                    } else {
                        return self.label_num.cmp(&other.label_num);
                    }
                }
            }
        }
    }
    impl PartialOrd for LayerNode {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(Ord::cmp(self, other))
        }
    }
    impl Eq for LayerNode {}
    impl PartialEq for LayerNode {
        fn eq(&self, other: &Self) -> bool {
            self.bfs_cons_num == other.bfs_cons_num
                && self.all_cons_num == other.all_cons_num
                && self.label_num == other.label_num
        }
    }

    #[inline]
    fn bfs_search_ordering<G>(
        g: G,
        node_idx: usize,
        order_idx: &mut usize,
        added: &mut Vec<bool>,
        matching_order: &mut Vec<usize>,
        node_con_num: &Vec<usize>,
        node_labels: &Vec<usize>,
        label_nums: &mut Vec<usize>,
    ) where
        G: IntoNeiborghbors + NodeIndexable + NodeCount,
    {
        let mut cur_order = *order_idx;
        let (mut start_of_layer, mut end_of_layer, mut last_added_pos) =
            (cur_order, cur_order, cur_order);
        let mut bfs_tree_node_cons = vec![0; g.node_count()];
        matching_order[cur_order] = node_idx;
        added[cur_order] = true;

        while cur_order <= last_added_pos {
            // add all current layer's nodes
            for order in start_of_layer..end_of_layer + 1 {
                let cur_node_idx = matching_order[order];
                for neighbor in g.neighbors(g.from_index(cur_node_idx)) {
                    let neighbor_id = g.to_index(neighbor);
                    bfs_tree_node_cons[neighbor_id] += 1;
                    if !added[neighbor_id] {
                        last_added_pos += 1;
                        added[last_added_pos] = true;
                        matching_order[last_added_pos] = neighbor_id;
                    }
                }
                cur_order += 1;
            }

            // reorder current layer based ordering rule
            let mut layer_ordering = BinaryHeap::new();
            for order in start_of_layer..end_of_layer + 1 {
                let cur_node_idx = matching_order[order];
                layer_ordering.push(LayerNode {
                    node_idx: cur_node_idx,
                    bfs_cons_num: bfs_tree_node_cons[cur_node_idx],
                    all_cons_num: node_con_num[cur_node_idx],
                    label_num: label_nums[node_labels[cur_node_idx]],
                });
            }
            let mut order_counter = start_of_layer;
            while let Some(layer_node) = layer_ordering.pop() {
                matching_order[order_counter] = layer_node.node_idx;
                label_nums[node_labels[layer_node.node_idx]] -= 1;
                order_counter += 1;
            }

            start_of_layer = end_of_layer + 1;
            end_of_layer = last_added_pos;
        }

        *order_idx = cur_order;
    }

    matching_order
}

#[allow(dead_code)]
/// init the nums of Rnew and Rinout for every node in the g0
fn init_r_new_inout<G>(
    g: G,
    order: &Vec<usize>,
    node_labels: &Vec<usize>,
    max_label: usize,
) -> (Vec<Vec<(usize, usize)>>, Vec<Vec<(usize, usize)>>)
where
    G: NodeCount + NodeIndexable + IntoNeiborghbors,
{
    // used to mark how many time a node has be visited, usize::MAX mean it already exist in the mapping
    let mut inner_node_cons = vec![0usize; g.node_count()];
    let mut r_new = vec![vec![]; max_label + 1];
    let mut r_in_out = vec![vec![]; max_label + 1];
    // `label_tmp` is used to record the label and num of same labels of current node's neighbor
    let mut label_tmp_in_out = vec![0; max_label + 1];
    let mut label_tmp_new = vec![0; max_label + 1];

    let max_order = order.len();
    let mut cur_order = 0;
    while max_order > cur_order {
        let cur_node_idx = order[cur_order];
        inner_node_cons[cur_node_idx] = usize::MAX;

        for neighbor in g.neighbors(g.from_index(cur_node_idx)) {
            let neighbor_id = g.to_index(neighbor);
            let neighbor_cons = inner_node_cons[neighbor_id];
            if neighbor_cons != usize::MAX && neighbor_cons > 0 {
                label_tmp_in_out[node_labels[neighbor_id]] += 1;
            } else if inner_node_cons[neighbor_id] == 0 {
                label_tmp_new[node_labels[neighbor_id]] += 1;
            }
        }

        for neighbor in g.neighbors(g.from_index(cur_node_idx)) {
            let neighbor_label = node_labels[g.to_index(neighbor)];
            if label_tmp_in_out[neighbor_label] > 0 {
                r_in_out[cur_node_idx].push((neighbor_label, label_tmp_in_out[neighbor_label]));
                label_tmp_in_out[neighbor_label] = 0;
            }
            if label_tmp_new[neighbor_label] > 0 {
                r_new[cur_node_idx].push((neighbor_label, label_tmp_new[neighbor_label]));
                label_tmp_new[neighbor_label] = 0;
            }

            // mark the unvisited neighbor nodes of current node as visited
            if inner_node_cons[g.to_index(neighbor)] != usize::MAX {
                inner_node_cons[g.to_index(neighbor)] += 1;
            }
        }

        cur_order += 1;
    }

    (r_new, r_in_out)
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------

/// check if g0 is isomorphic with a subgraph of g1
pub fn vf2pp_isomorphsim_label_subgraph_matching<G0, G1, F0, F1, NM, EM>(
    g0: G0,
    g1: G1,
    label0: F0,
    label1: F1,
    node_matcher: Option<NM>,
    edge_matcher: Option<EM>,
) -> bool
where
    G0: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    G1: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    F0: FnMut(G0::NodeWeight) -> usize,
    F1: FnMut(G1::NodeWeight) -> usize,
    NM: FnMut(G0::NodeWeight, G1::NodeWeight) -> bool,
    EM: FnMut(G0::EdgeWeight, G1::EdgeWeight) -> bool,
{
    todo!()
}

/// matching without using label
pub fn vf2pp_isomorphsim_matching<G0, G1>(g0: G0, g1: G1) -> bool
where
    G0: IntoNeighborsDirected + NodeIndexable,
    G1: IntoNeighborsDirected + NodeIndexable,
{
    todo!()
}

/// matching without using label, matching subgraph
pub fn vf2pp_isomorphsim_subgraph_matching<G0, G1>(g0: G0, g1: G1) -> bool
where
    G0: IntoNeighborsDirected + NodeIndexable,
    G1: IntoNeighborsDirected + NodeIndexable,
{
    todo!()
}

/// matching without using label, but supports semantic check
pub fn vf2pp_isomorphsim_semantic_matching<G0, G1, NM, EM>(
    g0: G0,
    g1: G1,
    node_matcher: Option<NM>,
    edge_matcher: Option<EM>,
) -> bool
where
    G0: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    G1: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    NM: FnMut(G0::NodeWeight, G1::NodeWeight) -> bool,
    EM: FnMut(G0::EdgeWeight, G1::EdgeWeight) -> bool,
{
    todo!()
}

/// matching without using label, but supports semantic check, matching subgraph of g1
pub fn vf2pp_isomorphsim_semantic_subgraph_matching<G0, G1, NM, EM>(
    g0: G0,
    g1: G1,
    node_matcher: Option<NM>,
    edge_matcher: Option<EM>,
) -> bool
where
    G0: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    G1: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    NM: FnMut(G0::NodeWeight, G1::NodeWeight) -> bool,
    EM: FnMut(G0::EdgeWeight, G1::EdgeWeight) -> bool,
{
    todo!()
}

/// a iterator return all possible match result of isomorphism
pub fn vf2pp_isomorphsim_label_subgraph_matching_iter<G0, G1, F0, F1, NM, EM>(
    g0: G0,
    g1: G1,
    label0: F0,
    label1: F1,
    node_matcher: Option<NM>,
    edge_matcher: Option<EM>,
) -> Option<GraphMatcher>
where
    G0: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    G1: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    F0: FnMut(G0::NodeWeight) -> usize,
    F1: FnMut(G1::NodeWeight) -> usize,
    NM: FnMut(G0::NodeWeight, G1::NodeWeight) -> bool,
    EM: FnMut(G0::EdgeWeight, G1::EdgeWeight) -> bool,
{
    todo!()
}

/// return the mapping result for matching result of isomorphism
pub fn vf2pp_isomorphsim_label_matching_mapping<G0, G1, F0, F1, NM, EM>(
    g0: G0,
    g1: G1,
    label0: F0,
    label1: F1,
    node_matcher: Option<NM>,
    edge_matcher: Option<EM>,
) -> Option<Vec<usize>>
where
    G0: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    G1: IntoNeighborsDirected + NodeIndexable + GraphDataAccess,
    F0: FnMut(G0::NodeWeight) -> usize,
    F1: FnMut(G1::NodeWeight) -> usize,
    NM: FnMut(G0::NodeWeight, G1::NodeWeight) -> bool,
    EM: FnMut(G0::EdgeWeight, G1::EdgeWeight) -> bool,
{
    todo!()
}

pub struct GraphMatcher {}
