use crate::graph::{
    visit::{
        GetAdjacencyMatrix, GraphDataAccess, IntoEdgeDirected, IntoNeighborsDirected, NodeCount,
        NodeIndexable,
    },
    Direction,
};
use std::collections::{HashMap, HashSet};

use self::semantic::{EdgeMatcher, NoSemanticMatch, NodeMatcher};

/// state space representation of VF2 algorithm
struct SSR<G: GetAdjacencyMatrix> {
    graph: G,
    // the node idx mapping to another graph, don't exist indicate no reflection
    mapping: Vec<usize>,
    // the out of Tout(s) nodes, zero indicates the current node position is not
    // in the Tout(s) range
    outs: Vec<usize>,
    // the in of Tins(s) nodes, zero indicates the current node position is not
    // in the Tout(s) range
    ins: Vec<usize>,
    // record the possible depth val possition of a specific depth except self,
    // the first param represent the out position, and sendcond represent the in
    // position. None indicates it doesn't have overlap position with other depth
    // position. Used to match possible candidate precisely.
    overlap_depth_record: HashMap<usize, (Vec<usize>, Vec<usize>)>,
    // node count of underlying graph
    node_count: usize,
    // current depth of search
    depth: usize,
    adjacency_matrix: G::AdjMatrix,
}

impl<G> SSR<G>
where
    G: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    fn new(g: G) -> Self {
        let node_count = g.node_count();
        Self {
            graph: g,
            mapping: vec![usize::MAX; node_count],
            outs: vec![0; node_count],
            ins: vec![0; node_count],
            overlap_depth_record: Default::default(),
            node_count,
            depth: 0,
            adjacency_matrix: g.adjacency_matrix(),
        }
    }
}

#[allow(dead_code)]
pub fn is_isomorphism_semantic_matching<G0, G1, F0, F1>(
    g0: G0,
    g1: G1,
    node_match: F0,
    edge_match: F1,
    match_subgraph: bool,
) -> bool
where
    G0: NodeIndexable
        + NodeCount
        + GetAdjacencyMatrix
        + IntoNeighborsDirected
        + GraphDataAccess
        + IntoEdgeDirected,
    G1: NodeIndexable
        + NodeCount
        + GetAdjacencyMatrix
        + IntoNeighborsDirected
        + GraphDataAccess
        + IntoEdgeDirected,
    F0: Fn(G0::NodeWeight, G1::NodeWeight) -> bool,
    F1: Fn(G0::EdgeWeight, G1::EdgeWeight) -> bool,
{
    isomorphism_semantic_matching_iter(g0, g1, node_match, edge_match, match_subgraph)
        .map(|mut iter| iter.next().is_some())
        .unwrap_or(false)
}

#[allow(dead_code)]
pub fn isomorphism_semantic_matching_iter<G0, G1, F0, F1>(
    g0: G0,
    g1: G1,
    node_matcher: F0,
    edge_matcher: F1,
    match_sub_graph: bool,
) -> Option<impl Iterator<Item = Vec<usize>>>
where
    G0: NodeIndexable
        + NodeCount
        + GetAdjacencyMatrix
        + IntoNeighborsDirected
        + GraphDataAccess
        + IntoEdgeDirected,
    G1: NodeIndexable
        + NodeCount
        + GetAdjacencyMatrix
        + IntoNeighborsDirected
        + GraphDataAccess
        + IntoEdgeDirected,
    F0: Fn(G0::NodeWeight, G1::NodeWeight) -> bool,
    F1: Fn(G0::EdgeWeight, G1::EdgeWeight) -> bool,
{
    if !match_sub_graph && g0.node_count() != g1.node_bound() {
        return None;
    }
    let (s0, s1) = (SSR::new(g0), SSR::new(g1));

    Some(IsomorphismMatcher {
        s0,
        s1,
        node_matcher,
        edge_matcher,
        stack: Default::default(),
        match_sub_graph,
        last_from: None,
    })
}

#[allow(dead_code)]
/// check if g0 is isomorphism to a subgraph of g1
/// Step 1: find candidate pair in the T0 ^ T1, we could use a queue to store all
/// nodes in g0, and pop one by one; For every node in g0, we need to iterate all
/// unmapped nodes in g1, if found, we need to pop next g0 node, if fail to match,
/// we need to match last candidate pair
/// Step 2: check the feasibility rules for candidate pair, if succeed, mark current
/// candidate pair in the mapping, if fail, pop up last mapping pair.
///
/// If match_sub_graph is true, check if g0 is isomorphism to a induced subgraph of g1,
/// and return iterator all of mapping pair for the isomorphism relation.
pub fn is_isomorphism_matching<G0, G1>(g0: G0, g1: G1, match_sub_graph: bool) -> bool
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    isomorphism_matching_iter(g0, g1, match_sub_graph)
        .as_mut()
        .map(|iter| iter.next().is_some())
        .unwrap_or(false)
}

#[allow(dead_code)]
pub fn isomorphism_matching_iter<G0, G1>(
    g0: G0,
    g1: G1,
    match_sub_graph: bool,
) -> Option<impl Iterator<Item = Vec<usize>>>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    if !match_sub_graph && g0.node_count() != g1.node_bound() {
        return None;
    }
    let (s0, s1) = (SSR::new(g0), SSR::new(g1));

    Some(IsomorphismMatcher {
        s0,
        s1,
        node_matcher: NoSemanticMatch,
        edge_matcher: NoSemanticMatch,
        stack: Default::default(),
        match_sub_graph,
        last_from: None,
    })
}

struct IsomorphismMatcher<G0, G1, NM, EM>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    NM: NodeMatcher<G0, G1>,
    EM: EdgeMatcher<G0, G1>,
{
    s0: SSR<G0>,
    s1: SSR<G1>,
    node_matcher: NM,
    edge_matcher: EM,
    stack: Vec<(usize, usize)>,
    match_sub_graph: bool,
    last_from: Option<usize>,
}

impl<G0, G1, NM, EM> Iterator for IsomorphismMatcher<G0, G1, NM, EM>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    NM: NodeMatcher<G0, G1>,
    EM: EdgeMatcher<G0, G1>,
{
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        isomorphsim_matching(
            &mut self.s0,
            &mut self.s1,
            &mut self.node_matcher,
            &mut self.edge_matcher,
            self.match_sub_graph,
            &mut self.stack,
            &mut self.last_from,
        )
    }
}

fn isomorphsim_matching<G0, G1, NM, EM>(
    s0: &mut SSR<G0>,
    s1: &mut SSR<G1>,
    node_matcher: &mut NM,
    edge_matcher: &mut EM,
    match_sub_graph: bool,
    stack: &mut Vec<(usize, usize)>,
    last_from: &mut Option<usize>,
) -> Option<Vec<usize>>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    NM: NodeMatcher<G0, G1>,
    EM: EdgeMatcher<G0, G1>,
{
    let mut s1_from = last_from.as_ref().map(|from| from.clone()).unwrap_or(0);

    'outer: loop {
        if let Some(nodes) = check_next_candidate_from(
            &s0,
            &s1,
            node_matcher,
            edge_matcher,
            match_sub_graph,
            s1_from,
        ) {
            mark_candidate_pair(s0, s1, nodes);
            s1_from = 0;
            stack.push(nodes);
            // println!("mark nodes: {} {}", nodes.0, nodes.1);
            continue;
        }

        // check if all nodes has been added to the mapping
        if s0.depth == s0.node_count {
            let result = Some(s0.mapping.iter().map(|v| v.clone()).collect());
            // in case empty graph, there is no stack to pop
            if let Some(last_pair) = stack.pop() {
                unmark_candidate_mapping(s0, s1, last_pair);
                *last_from = Some(last_pair.1 + 1);
            }
            return result;
        }

        // pop last node mapping pair, and pick the node pair from last position
        if let Some(last_nodes_pair) = stack.pop() {
            // println!("unmark nodes: {} {}", last_nodes_pair.0, last_nodes_pair.1);
            unmark_candidate_mapping(s0, s1, last_nodes_pair);
            // continue to search the candidate for last_nodes_pair.0 in g0, start from
            // last_nodes_pair.1 + 1
            s1_from = last_nodes_pair.1 + 1;
            continue;
        }

        // we could find a mapping in the stack, which indicates there is not
        // isomorphism relation between the g0 and g1
        break 'outer;
    }

    None
}

#[allow(dead_code)]
/// Add a feasible pair node to the mapping
fn mark_candidate_pair<G0, G1>(s0: &mut SSR<G0>, s1: &mut SSR<G1>, (n0, n1): (usize, usize))
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    #[inline]
    fn advance_ssr<G>(s: &mut SSR<G>, (n0, n1): (usize, usize))
    where
        G: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    {
        debug_assert!(s.mapping[n0] == usize::MAX);
        s.mapping[n0] = n1;
        s.depth += 1;
        for out_neighbor in s
            .graph
            .neighbors_directed(s.graph.from_index(n0), Direction::Outcoming)
        {
            let out_neighbor_id = s.graph.to_index(out_neighbor);
            if s.outs[out_neighbor_id] == 0 {
                s.outs[out_neighbor_id] = s.depth;
            } else {
                let overlap_record = s
                    .overlap_depth_record
                    .entry(s.depth)
                    .or_insert_with(|| Default::default());
                overlap_record.0.push(out_neighbor_id);
            }
        }

        for in_neighbor in s
            .graph
            .neighbors_directed(s.graph.from_index(n0), Direction::Incoming)
        {
            let in_neighbor_id = s.graph.to_index(in_neighbor);
            if s.ins[in_neighbor_id] == 0 {
                s.ins[in_neighbor_id] = s.depth;
            } else {
                let overlap_record = s
                    .overlap_depth_record
                    .entry(s.depth)
                    .or_insert_with(|| Default::default());
                overlap_record.1.push(in_neighbor_id);
            }
        }
    }

    advance_ssr(s0, (n0, n1));
    advance_ssr(s1, (n1, n0));
}

#[allow(dead_code)]
/// unmark a pair of node from the mapping
fn unmark_candidate_mapping<G0, G1>(s0: &mut SSR<G0>, s1: &mut SSR<G1>, (n0, n1): (usize, usize))
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    #[inline]
    fn backforward_ssr_state<G>(s: &mut SSR<G>, n: usize)
    where
        G: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    {
        debug_assert_ne!(s.mapping[n], usize::MAX);
        s.mapping[n] = usize::MAX;

        let depth = s.depth;
        for d in s.ins.iter_mut() {
            if *d == depth {
                *d = 0;
            }
        }
        for d in s.outs.iter_mut() {
            if *d == depth {
                *d = 0;
            }
        }

        s.overlap_depth_record.remove(&depth);
        s.depth -= 1;
    }

    backforward_ssr_state(s0, n0);
    backforward_ssr_state(s1, n1);
}

#[allow(dead_code)]
/// check the feasibility rules of the possible node pair,
/// There are five rules for standard VF2 algorithm:
/// Rpre ^ Rssuc ^ Rin ^ Rout ^ Rnew
fn check_feasibility<G0, G1, NM, EM>(
    s0: &SSR<G0>,
    s1: &SSR<G1>,
    (n0, n1): (usize, usize),
    node_matcher: &mut NM,
    edge_matcher: &mut EM,
    match_sub_graph: bool,
) -> bool
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    NM: NodeMatcher<G0, G1>,
    EM: EdgeMatcher<G0, G1>,
{
    let s0_node_id = s0.graph.from_index(n0);
    let s1_node_id = s1.graph.from_index(n1);

    // check the node semantice feasibility
    if NM::enabled() && !node_matcher.eq(s0.graph, s1.graph, s0_node_id, s1_node_id) {
        return false;
    }

    macro_rules! field {
        ($val: tt, 0) => {
            $val.0
        };
        ($val: tt, 1) => {
            $val.1
        };
        ($val: tt, 1 - 0) => {
            $val.1
        };
        ($val: tt, 1 - 1) => {
            $val.0
        };
    }

    macro_rules! check_r_succ_prev {
        ($s0_node_id: tt, $s1_node_id: tt, $s0: tt, $s1: tt, $start_point: tt) => {{
            let (
                mut s_counter_sum,
                mut s_succ_t_ins,
                mut s_succ_t_outs,
                mut s_succ_outside,
                mut s_prev_t_ins,
                mut s_prev_t_outs,
                mut s_prev_outside,
            ) = (0, 0, 0, 0, 0, 0, 0);

            let s0_node_idx = $s0.graph.to_index($s0_node_id);
            for out_neighbor in $s0
                .graph
                .neighbors_directed($s0_node_id, Direction::Outcoming)
            {
                let out_neighrbor_idx = $s0.graph.to_index(out_neighbor);
                // if this is a self-loop edge, then the n1 must be self-loop too
                let mapping_in_g1 = if out_neighrbor_idx == s0_node_idx {
                    $s1.graph.to_index($s1_node_id)
                } else {
                    $s0.mapping[out_neighrbor_idx]
                };
                let is_in_mapping = mapping_in_g1 != usize::MAX;
                // check Rpre rule
                if is_in_mapping {
                    // check mapping position is the outcoming neighbor of n1
                    if !$s1.graph.is_adjacent(
                        &$s1.adjacency_matrix,
                        $s1_node_id,
                        $s1.graph.from_index(mapping_in_g1),
                    ) {
                        return false;
                    }

                    // check the semantic rule of edge
                    if EM::enabled() {
                        let matcher_nodes_pair = (($s0_node_id, out_neighbor), ($s1_node_id, $s1.graph.from_index(mapping_in_g1)));
                        if !edge_matcher.eq(
                                s0.graph,
                                s1.graph,
                                field!(matcher_nodes_pair, $start_point),
                                field!(matcher_nodes_pair, 1 - $start_point),
                            ) {
                                return false;
                            }

                    }
                } else {
                    let neighbor_in_outs = $s0.outs[out_neighrbor_idx] != 0;
                    let neighbor_in_ins = $s0.ins[out_neighrbor_idx] != 0;
                    if neighbor_in_outs {
                        s_succ_t_outs += 1;
                    }
                    if neighbor_in_ins {
                        s_succ_t_ins += 1;
                    }
                    if !neighbor_in_outs && !neighbor_in_ins {
                        s_succ_outside += 1;
                    }
                }

                s_counter_sum += 1;
            }

            for in_neighbor in $s0
                .graph
                .neighbors_directed($s0_node_id, Direction::Incoming)
            {
                let in_neighbor_idx = $s0.graph.to_index(in_neighbor);
                let mapping_in_g1 = $s0.mapping[in_neighbor_idx];
                // we already checked the self-loop case in outcoming edge,
                // so we could ignore that case here
                let is_in_maping = mapping_in_g1 != usize::MAX;
                if is_in_maping {
                    // check Rsucc rule
                    // check mapping position is the outcoming neighbor of n1
                    if !$s1.graph.is_adjacent(
                        &$s1.adjacency_matrix,
                        $s1.graph.from_index(mapping_in_g1),
                        $s1_node_id,
                    ) {
                        return false;
                    }

                    // get the edge_id beteween $s0_node_id and in_neighbor, we have two options,
                    // one: definie a new trait, which require the graph to not only return the neighbor_id
                    // but also the edge_id, but that seems unneccsary for nosemantic matching.
                    // two: define a independent trait, which read the two node_id pair, and return edge_id if
                    // exists. The drawback of this method is we need to iterate all the edges list every time,
                    // very low performance, but the code may seem more clear.

                    // check the semantic rule of edge
                    if EM::enabled() {
                        let matcher_nodes_pair = ((in_neighbor, $s0_node_id), ($s1.graph.from_index(mapping_in_g1), $s1_node_id));
                        if !edge_matcher.eq(
                            s0.graph,
                            s1.graph,
                            field!(matcher_nodes_pair, $start_point),
                            field!(matcher_nodes_pair, 1 - $start_point),
                        ) {
                            return false;
                        }
                    }
                } else {
                    let neighbor_in_outs = $s0.outs[in_neighbor_idx] != 0;
                    let neighbor_in_ins = $s0.ins[in_neighbor_idx] != 0;
                    if neighbor_in_outs {
                        s_prev_t_outs += 1;
                    }
                    if neighbor_in_ins {
                        s_prev_t_ins += 1;
                    }

                    if !neighbor_in_ins && !neighbor_in_outs {
                        s_prev_outside += 1;
                    }
                }

                s_counter_sum += 1;
            }

            (
                s_counter_sum,
                s_succ_t_ins,
                s_succ_t_outs,
                s_succ_outside,
                s_prev_t_ins,
                s_prev_t_outs,
                s_prev_outside,
            )
        }};
    }

    let (
        s0_counter_sum,
        s0_succ_t_ins,
        s0_succ_t_outs,
        s0_succ_outside,
        s0_prev_t_ins,
        s0_prev_t_outs,
        s0_prev_outside,
    ) = check_r_succ_prev!(s0_node_id, s1_node_id, s0, s1, 0);
    let (
        s1_counter_sum,
        s1_succ_t_ins,
        s1_succ_t_outs,
        s1_succ_outside,
        s1_prev_t_ins,
        s1_prev_t_outs,
        s1_prev_outside,
    ) = check_r_succ_prev!(s1_node_id, s0_node_id, s1, s0, 1);

    if match_sub_graph {
        // check Rin rule
        s0_succ_t_ins <= s1_succ_t_ins
            && s0_prev_t_ins <= s1_prev_t_ins
            // check Rout rule
            && s0_succ_t_outs <= s1_succ_t_outs
            && s0_prev_t_outs <= s1_prev_t_outs
            // check Rnew rule
            && s0_succ_outside <= s1_succ_outside
            && s0_prev_outside <= s1_prev_outside
            // check sum
            && s0_counter_sum <= s1_counter_sum
    } else {
        // check Rin rule
        s0_succ_t_ins == s1_succ_t_ins
            && s0_prev_t_ins == s1_prev_t_ins
            // check Rout rule
            && s0_succ_t_outs == s1_succ_t_outs
            && s0_prev_t_outs == s1_prev_t_outs
            // check Rnew rule
            && s0_succ_outside == s1_succ_outside
            && s0_prev_outside == s1_prev_outside
            // check sum
            && s0_counter_sum == s1_counter_sum
    }
}

#[allow(dead_code)]
/// Pick a possible next node pair from ssr pair, if we check it with depth, that means
/// we need to scan the whole array to find correct position, it will cause much performance
/// cost, the best way is to scan from the beginning, pick the first one, so we could
/// record the position of the minimum value so we could skip some scan every time with
/// little space cost.
/// And, we need to check if the depth of the candidate match.
static TOUT: u8 = 0;
static TIN: u8 = 1;

fn candidate_iter_from<'a, G0, G1>(
    s0: &'a SSR<G0>,
    s1: &'a SSR<G1>,
    s1_from: usize,
) -> Box<dyn Iterator<Item = (usize, usize)> + 'a>
where
    G0: GetAdjacencyMatrix,
    G1: GetAdjacencyMatrix,
{
    // check possible Tout or Tin candidate pair
    for t in [TOUT, TIN] {
        let s0_outs_or_ins = if t == TOUT { &s0.outs } else { &s0.ins };

        let s0_out_or_in = s0_outs_or_ins[0..]
            .iter()
            .enumerate()
            .find(|(i, &depth)| depth != 0 && s0.mapping[*i] == usize::MAX)
            .map(|(i, &depth)| (depth, i));

        if let Some((depth, s0_node_id)) = s0_out_or_in {
            let (s1_min_start, s1_outs_or_ins) = if t == TOUT {
                (s1_from, &s1.outs)
            } else {
                (s1_from, &s1.ins)
            };

            let overlap_depth_list: HashSet<usize> = s1
                .overlap_depth_record
                .get(&depth)
                .map(|(outs, ins)| {
                    if t == TOUT {
                        outs.iter().map(|i| i.clone()).collect()
                    } else {
                        ins.iter().map(|i| i.clone()).collect()
                    }
                })
                .unwrap_or(Default::default());
            let s1_out_or_in_iter = s1_outs_or_ins[s1_min_start..]
                .iter()
                .enumerate()
                .filter(move |(i, &d)| {
                    let node_idx = i + s1_min_start;
                    if d > depth || s1.mapping[node_idx] != usize::MAX {
                        return false;
                    }

                    // check if it exist in outs ins or in the upper depth
                    d == depth || overlap_depth_list.contains(&node_idx)
                })
                .map(move |(i, _)| (s0_node_id, i + s1_min_start));

            return Box::new(s1_out_or_in_iter);
        }
    }

    // there is no possible outs or ins, maybe we need to proceed from unknown area
    if let Some(s0_node_id) = s0.mapping.iter().position(|&m| m == usize::MAX) {
        return Box::new(
            s1.mapping[s1_from..]
                .iter()
                .enumerate()
                .filter(|(_, &p)| p == usize::MAX)
                .map(move |(i, _)| (s0_node_id, i + s1_from)),
        );
    }

    Box::new(([] as [(usize, usize); 0]).into_iter())
}

fn check_next_candidate_from<G0, G1, NM, EM>(
    s0: &SSR<G0>,
    s1: &SSR<G1>,
    node_matcher: &mut NM,
    edge_matcher: &mut EM,
    match_sub_graph: bool,
    from: usize,
) -> Option<(usize, usize)>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    NM: NodeMatcher<G0, G1>,
    EM: EdgeMatcher<G0, G1>,
{
    let mut candidate_node_iter = candidate_iter_from(s0, s1, from);
    // try all possible candidate from n0 in g0 to every node in g1
    while let Some(nodes) = candidate_node_iter.next() {
        let is_feasible =
            check_feasibility(s0, s1, nodes, node_matcher, edge_matcher, match_sub_graph);
        if is_feasible {
            return Some(nodes);
        }
    }

    None
}

#[allow(dead_code)]
/// pick one candidate node pair and check if it's feasible, return Some((usize, usize)) if
/// it's feasible, None represents infeasible
fn check_next_candidate<G0, G1, NM, EM>(
    s0: &SSR<G0>,
    s1: &SSR<G1>,
    node_matcher: &mut NM,
    edge_matcher: &mut EM,
    match_sub_graph: bool,
) -> Option<(usize, usize)>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    NM: NodeMatcher<G0, G1>,
    EM: EdgeMatcher<G0, G1>,
{
    check_next_candidate_from(s0, s1, node_matcher, edge_matcher, match_sub_graph, 0)
}

mod semantic {
    use crate::graph::{
        visit::{GraphBase, GraphDataAccess, IntoEdgeDirected},
        Direction,
    };

    pub trait NodeMatcher<G0: GraphBase, G1: GraphBase> {
        fn enabled() -> bool;

        fn eq(&mut self, g0: G0, g1: G1, node_id_0: G0::NodeId, node_id_1: G1::NodeId) -> bool;
    }

    pub trait EdgeMatcher<G0: GraphBase, G1: GraphBase> {
        fn enabled() -> bool;

        fn eq(
            &mut self,
            g0: G0,
            g1: G1,
            g0_nodes: (G0::NodeId, G0::NodeId),
            g1_nodes: (G1::NodeId, G1::NodeId),
        ) -> bool;
    }

    pub struct NoSemanticMatch;

    impl<G0: GraphBase, G1: GraphBase> NodeMatcher<G0, G1> for NoSemanticMatch {
        fn enabled() -> bool {
            false
        }

        fn eq(
            &mut self,
            _g0: G0,
            _g1: G1,
            _node_id_0: <G0 as GraphBase>::NodeId,
            _node_id_1: <G1 as GraphBase>::NodeId,
        ) -> bool {
            false
        }
    }

    impl<G0: GraphBase, G1: GraphBase> EdgeMatcher<G0, G1> for NoSemanticMatch {
        fn enabled() -> bool {
            false
        }

        fn eq(
            &mut self,
            _g0: G0,
            _g1: G1,
            _: (G0::NodeId, G0::NodeId),
            _: (G1::NodeId, G1::NodeId),
        ) -> bool {
            false
        }
    }

    impl<G0, G1, F> NodeMatcher<G0, G1> for F
    where
        G0: GraphDataAccess + IntoEdgeDirected,
        G1: GraphDataAccess + IntoEdgeDirected,
        F: FnMut(G0::NodeWeight, G1::NodeWeight) -> bool,
    {
        fn enabled() -> bool {
            true
        }

        fn eq(
            &mut self,
            g0: G0,
            g1: G1,
            node_id_0: <G0 as GraphBase>::NodeId,
            node_id_1: <G1 as GraphBase>::NodeId,
        ) -> bool {
            if let (Some(n0), Some(n1)) = (
                G0::node_weight(g0, node_id_0),
                G1::node_weight(g1, node_id_1),
            ) {
                self(n0, n1)
            } else {
                false
            }
        }
    }

    impl<G0, G1, F> EdgeMatcher<G0, G1> for F
    where
        G0: GraphDataAccess + IntoEdgeDirected + Copy,
        G1: GraphDataAccess + IntoEdgeDirected + Copy,
        F: FnMut(G0::EdgeWeight, G1::EdgeWeight) -> bool,
    {
        fn enabled() -> bool {
            true
        }

        fn eq(
            &mut self,
            g0: G0,
            g1: G1,
            (g0_node_id_0, g0_node_id_1): (<G0 as GraphBase>::NodeId, <G0 as GraphBase>::NodeId),
            (g1_node_id_0, g1_node_id_1): (<G1 as GraphBase>::NodeId, <G1 as GraphBase>::NodeId),
        ) -> bool {
            let edge_0: Option<G0::EdgeId> =
                g0.edge_directed(g0_node_id_0, g0_node_id_1, Direction::Outcoming);
            let edge_1: Option<G1::EdgeId> =
                g1.edge_directed(g1_node_id_0, g1_node_id_1, Direction::Outcoming);

            if edge_0.is_none() && edge_1.is_none() {
                return false;
            }

            if let (Some(n0), Some(n1)) = (
                G0::edge_weight(g0, edge_0.unwrap()),
                G1::edge_weight(g1, edge_1.unwrap()),
            ) {
                self(n0, n1)
            } else {
                false
            }
        }
    }
}
