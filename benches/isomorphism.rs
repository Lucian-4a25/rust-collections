#![feature(test)]
extern crate test;

use test::Bencher;

#[allow(dead_code)]
mod common;
use common::*;

use rust_collections::graph::algo::isomorphism::is_isomorphism_matching;

#[bench]
fn petersen_iso_bench(bench: &mut Bencher) {
    let a = digraph().petersen_a();
    let b = digraph().petersen_b();

    bench.iter(|| is_isomorphism_matching(&a, &b, false));
    assert!(is_isomorphism_matching(&a, &b, false));
}

#[bench]
fn petersen_undir_iso_bench(bench: &mut Bencher) {
    let a = ungraph().petersen_a();
    let b = ungraph().petersen_b();

    bench.iter(|| is_isomorphism_matching(&a, &b, false));
    assert!(is_isomorphism_matching(&a, &b, false));
}

#[bench]
fn full_iso_bench(bench: &mut Bencher) {
    let a = ungraph().full_a();
    let b = ungraph().full_b();

    bench.iter(|| is_isomorphism_matching(&a, &b, false));
    assert!(is_isomorphism_matching(&a, &b, false));
}

#[bench]
fn praust_dir_no_iso_bench(bench: &mut Bencher) {
    let a = digraph().praust_a();
    let b = digraph().praust_b();

    bench.iter(|| is_isomorphism_matching(&a, &b, false));
    assert!(!is_isomorphism_matching(&a, &b, false));
}

#[bench]
fn praust_undir_no_iso_bench(bench: &mut Bencher) {
    let a = ungraph().praust_a();
    let b = ungraph().praust_b();

    bench.iter(|| is_isomorphism_matching(&a, &b, false));
    assert!(!is_isomorphism_matching(&a, &b, false));
}

#[bench]
fn iso_large(bench: &mut Bencher) {
    let g0 = graph_from_file("benches/res/graph_1000n_1000e.txt");
    let g1 = graph_from_file("benches/res/graph_1000n_1000e.txt");

    bench.iter(|| is_isomorphism_matching(&g0, &g1, false));
    assert!(is_isomorphism_matching(&g0, &g1, false));
}

const SIZE: usize = 10;

#[bench]
fn test_realloc(bench: &mut Bencher) {
    bench.iter(|| vec![0usize; SIZE]);
}

#[bench]
fn test_reset(bench: &mut Bencher) {
    let mut v: Vec<usize> = vec![100; SIZE];
    bench.iter(|| {
        for i in 0..SIZE {
            v[i] = 0;
        }
    });
}
