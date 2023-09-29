use crate::graph::{
    visit::{GetAdjacencyMatrix, IntoNeighborsDirected, NodeCount, NodeIndexable},
    Direction,
};
use std::collections::HashMap;

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
/// check if two graphs are isomorphism, two graphs are isomorphism when there are bijective between them,
/// and every pair of nodes' edges in each graph must be same nodes.
pub fn is_isomorphism_matching<G0, G1>(g0: G0, g1: G1) -> bool
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    isomorphism_matching_iter(g0, g1).is_some()
}

#[allow(dead_code)]
pub fn isomorphism_matching_iter<G0, G1>(g0: G0, g1: G1) -> Option<Vec<usize>>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    if g0.node_count() != g1.node_bound() {
        return None;
    }
    let (s0, s1) = (SSR::new(g0), SSR::new(g1));

    isomorphsim_matching(s0, s1, false)
}

#[allow(dead_code)]
/// check if g0 is isomorphism to a subgraph of g1
/// Step 1: find candidate pair in the T0 ^ T1, we could use a queue to store all
/// nodes in g0, and pop one by one; For every node in g0, we need to iterate all
/// unmapped nodes in g1, if found, we need to pop next g0 node, if fail to match,
/// we need to match last candidate pair
/// Step 2: check the feasibility rules for candidate pair, if succeed, mark current
/// candidate pair in the mapping, if fail, pop up last mapping pair.
pub fn is_subgraph_isomorphism<G0, G1>(g0: G0, g1: G1) -> bool
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    subgraph_isomorphism_matching_iter(g0, g1).is_some()
}

#[allow(dead_code)]
/// check if g0 is isomorphism to a induced subgraph of g1, and return iterator all of mapping
/// pair for the isomorphism relation.
pub fn subgraph_isomorphism_matching_iter<G0, G1>(g0: G0, g1: G1) -> Option<Vec<usize>>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    let (s0, s1) = (SSR::new(g0), SSR::new(g1));
    isomorphsim_matching(s0, s1, true)
}

fn isomorphsim_matching<G0, G1>(
    mut s0: SSR<G0>,
    mut s1: SSR<G1>,
    match_sub_graph: bool,
) -> Option<Vec<usize>>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    let mut stack = Vec::new();

    'outer: loop {
        let res = check_next_candidate(&s1, &s1, match_sub_graph);
        if let Some(nodes) = res {
            mark_candidate_pair(&mut s0, &mut s1, nodes);
            stack.push(nodes);
            continue;
        }

        // check if all nodes has been added to the mapping
        if s0.depth == s0.node_count {
            return Some(s0.mapping);
        }

        // pop last node mapping pair, and pick the node pair from last position
        while let Some(last_nodes_pair) = stack.pop() {
            unmark_candidate_mapping(&mut s0, &mut s1, last_nodes_pair);
            // continue to search the candidate for last_nodes_pair.0 in g0, start from
            // last_nodes_pair.1 + 1
            if let Some(nodes) =
                check_next_candidate_from(&s0, &s1, last_nodes_pair.1 + 1, match_sub_graph)
            {
                stack.push(nodes);
                mark_candidate_pair(&mut s0, &mut s1, nodes);
                continue 'outer;
            }
        }

        // we could find a mapping in the stack, which indicates there is not
        // isomorphism relation between the g0 and g1
        break 'outer;
    }

    if s0.depth == s0.node_count {
        Some(s0.mapping)
    } else {
        None
    }
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
                    .entry(n0)
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
                    .entry(n0)
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
fn check_feasibility<G0, G1>(
    s0: &SSR<G0>,
    s1: &SSR<G1>,
    (n0, n1): (usize, usize),
    match_sub_graph: bool,
) -> bool
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    let s0_node_id = s0.graph.from_index(n0);
    let s1_node_id = s1.graph.from_index(n1);

    macro_rules! check_r_succ_prev {
        ($s0_node_id: tt, $s1_node_id: tt, $s0: tt, $s1: tt) => {{
            let (
                mut s_succ_counter,
                mut s_succ_t_ins,
                mut s_succ_t_outs,
                mut s_succ_empty,
                mut s_prev_t_ins,
                mut s_prev_t_outs,
                mut s_prev_outside,
            ) = (0, 0, 0, 0, 0, 0, 0);

            for out_neighbor in $s0
                .graph
                .neighbors_directed($s0_node_id, Direction::Outcoming)
            {
                let out_neighrbor_idx = $s0.graph.to_index(out_neighbor);
                let mapping_in_g1 = $s0.mapping[out_neighrbor_idx];
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
                }

                // if this is a self-loop edge, then the n1 must be self-loop too
                if !is_in_mapping {
                    let neighbor_in_outs = s0.outs[out_neighrbor_idx] != 0;
                    let neighbor_in_ins = s0.ins[out_neighrbor_idx] != 0;
                    if neighbor_in_outs {
                        s_succ_t_outs += 1;
                    }
                    if neighbor_in_ins {
                        s_succ_t_ins += 1;
                    }
                    if !neighbor_in_outs && !neighbor_in_ins {
                        s_succ_empty += 1;
                    }
                }

                s_succ_counter += 1;
            }

            for in_neighbor in $s0
                .graph
                .neighbors_directed($s0_node_id, Direction::Incoming)
            {
                let in_neighbor_idx = $s0.graph.to_index(in_neighbor);
                let mapping_in_g1 = $s0.mapping[in_neighbor_idx];
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
                }

                // if this is a self-loop edge, then the n1 must be self-loop too
                if !is_in_maping {
                    let neighbor_in_outs = s0.outs[in_neighbor_idx] != 0;
                    let neighbor_in_ins = s0.ins[in_neighbor_idx] != 0;
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

                s_succ_counter += 1;
            }

            (
                s_succ_counter,
                s_succ_t_ins,
                s_succ_t_outs,
                s_succ_empty,
                s_prev_t_ins,
                s_prev_t_outs,
                s_prev_outside,
            )
        }};
    }

    let (
        s0_succ_counter,
        s0_succ_t_ins,
        s0_succ_t_outs,
        s0_succ_empty,
        s0_prev_t_ins,
        s0_prev_t_outs,
        s0_prev_outside,
    ) = check_r_succ_prev!(s0_node_id, s1_node_id, s0, s1);
    let (
        s1_succ_counter,
        s1_succ_t_ins,
        s1_succ_t_outs,
        s1_succ_empty,
        s1_prev_t_ins,
        s1_prev_t_outs,
        s1_prev_outside,
    ) = check_r_succ_prev!(s1_node_id, s0_node_id, s1, s0);

    if match_sub_graph {
        // check Rin rule
        s0_succ_t_ins <= s1_succ_t_ins
            && s0_prev_t_ins <= s1_prev_t_ins
            // check Rout rule
            && s0_succ_t_outs <= s1_succ_t_outs
            && s0_prev_t_outs <= s1_prev_t_outs
            // check Rnew rule
            && s0_succ_empty <= s1_succ_empty
            && s0_prev_outside <= s1_prev_outside
            // check sum
            && s0_succ_counter <= s1_succ_counter
    } else {
        // check Rin rule
        s0_succ_t_ins == s1_succ_t_ins
            && s0_prev_t_ins == s1_prev_t_ins
            // check Rout rule
            && s0_succ_t_outs == s1_succ_t_outs
            && s0_prev_t_outs == s1_prev_t_outs
            // check Rnew rule
            && s0_succ_empty == s1_succ_empty
            && s0_prev_outside == s1_prev_outside
            // check sum
            && s0_succ_counter == s1_succ_counter
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

            let s1_out_or_in_iter = s1_outs_or_ins[s1_min_start..]
                .iter()
                .enumerate()
                .filter(move |(i, &d)| {
                    let node_idx = i + s1_min_start;
                    if d > depth || s1.mapping[node_idx] != usize::MAX {
                        return false;
                    }

                    // check if it exist in the upper depth
                    d == depth
                        || s1
                            .overlap_depth_record
                            .get(&node_idx)
                            .map(|(outs, ins)| {
                                if t == TOUT {
                                    outs.contains(&node_idx)
                                } else {
                                    ins.contains(&node_idx)
                                }
                            })
                            .unwrap_or(false)
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
                .filter(|&p| *p == usize::MAX)
                .map(move |i| (s0_node_id, i + s1_from)),
        );
    }

    Box::new(([] as [(usize, usize); 0]).into_iter())
}

fn check_next_candidate_from<G0, G1>(
    s0: &SSR<G0>,
    s1: &SSR<G1>,
    from: usize,
    match_sub_graph: bool,
) -> Option<(usize, usize)>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    let mut candidate_node_iter = candidate_iter_from(s0, s1, from);
    // try all possible candidate from n0 in g0 to every node in g1
    while let Some(nodes) = candidate_node_iter.next() {
        let is_feasible = check_feasibility(s0, s1, nodes, match_sub_graph);
        if is_feasible {
            return Some(nodes);
        }
    }

    None
}

/// pick one candidate node pair and check if it's feasible, return Some((usize, usize)) if
/// it's feasible, None represents infeasible
fn check_next_candidate<G0, G1>(
    s0: &SSR<G0>,
    s1: &SSR<G1>,
    match_sub_graph: bool,
) -> Option<(usize, usize)>
where
    G0: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
    G1: NodeIndexable + NodeCount + GetAdjacencyMatrix + IntoNeighborsDirected,
{
    check_next_candidate_from(s0, s1, 0, match_sub_graph)
}
