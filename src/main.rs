#![feature(dropck_eyepatch)]
#![feature(exclusive_range_pattern)]
#![feature(const_trait_impl)]
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(strict_provenance)]
#![feature(associated_type_defaults)]
#![feature(linked_list_remove)]

extern crate alloc;

use core::num;
use std::io::Write;
use std::mem::ManuallyDrop;
use std::thread;
use std::{
    alloc::Layout,
    collections::{BTreeMap, HashMap, LinkedList, VecDeque},
    fs::File,
    io::{Error, Read, Seek},
    mem::MaybeUninit,
    ptr::addr_of_mut,
    time::Duration,
};

use petgraph::{Directed, Direction};

// use crate::util::print_memory_usage;

// extern crate ptr;

mod binary_heap_cus;
mod btreemap_cus;
mod graph;
mod linked_list_cus;
pub mod raw_vec;
mod variance;
mod vec_cus;
mod vecdeque_cus;

fn main() {
    // vec::vec_usage();
    // vec_usage::vec_slice();
    // Box::leak(Box::new(c));

    // silly_default([3; 9]);
    // check_btree_memory();
    // binary_heap_usage();
    // check_stablegraph_memory_usage();
    // check_normalgraph_memory_usage();
    check_maybeunint_leaks();
}

/// be careful to use MaybeUninit::write, it will lead to memory leak
fn check_maybeunint_leaks() {
    let mut vals = vec![];
    for _ in 0..1000000 {
        // let _ = MaybeUninit::new("123".to_string());
        // let _ = ManuallyDrop::new("123".to_string());
        // let mut tmp = MaybeUninit::uninit() as MaybeUninit<String>;
        // tmp.write("123".to_string());
        // vals.push(tmp);

        let mut t = MaybeUninit::new("123".to_string());
        t.write("123".to_string());
        vals.push(t);
    }
}

#[test]
fn graph_csr_usage() {
    use petgraph::csr::Csr;

    let mut csr: Csr<_, _, Directed> = Csr::new();

    let famous_physicist = [
        "Galileo", "Newton", "Faraday", "Maxwell", "Einstein", "Planck",
    ];
    for name in famous_physicist.clone() {
        csr.add_node(name);
    }

    csr.add_edge(1u32.into(), 2u32.into(), 2);
}

#[test]
fn graph_matrix_usage() {
    use petgraph::matrix_graph::DiMatrix;

    let mut mg: _ = DiMatrix::new();
    let famous_physicist = [
        "Galileo", "Newton", "Faraday", "Maxwell", "Einstein", "Planck",
    ];
    for name in famous_physicist.clone() {
        mg.add_node(name);
    }

    mg.add_edge(1.into(), 2.into(), 2);
}

#[test]
fn graph_map_usage() {
    use petgraph::graphmap::GraphMap;
    use petgraph::Directed;

    let mut gm: GraphMap<_, _, Directed> = GraphMap::new();

    let famous_physicist = [
        "Galileo", "Newton", "Faraday", "Maxwell", "Einstein", "Planck",
    ];
    for name in famous_physicist.clone() {
        gm.add_node(name);
    }

    gm.add_edge("Galileo".into(), "Newton".into(), 2);
}

#[test]
fn itereate_graph() {
    use petgraph::graph::Graph;
    use petgraph::visit::Dfs;

    let mut graph = Graph::new();
    let famous_physicist = [
        "Galileo", "Newton", "Faraday", "Maxwell", "Einstein", "Planck",
    ];
    for name in famous_physicist.clone() {
        graph.add_node(name);
    }

    graph.add_edge(0.into(), 1.into(), 7);
    graph.add_edge(1.into(), 2.into(), 1);
    graph.add_edge(2.into(), 3.into(), 2);
    graph.add_edge(3.into(), 4.into(), 3);
    graph.add_edge(0.into(), 3.into(), 4);
    graph.add_edge(1.into(), 4.into(), 5);
    graph.add_edge(1.into(), 5.into(), 6);
    let mut dfs = Dfs::new(&graph, 0.into());

    while let Some(id) = dfs.next(&graph) {
        println!("node idx: {:?}", id);
    }
}

#[allow(dead_code)]
/// we use this method to compare the memory of Graph<Option<T>> and Graph<T>
fn check_normalgraph_memory_usage() {
    use petgraph::graph::Graph;

    struct Data {
        id: u64,
    }
    let mut graph: Graph<Data, usize> = Graph::new();
    let insertion_num = 3_000_000;
    for _ in 0..insertion_num {
        graph.add_node(Data {
            id: Default::default(),
        });
    }
}

#[allow(dead_code)]
/// we use this method to compare the memory of Graph<Option<T>> and Graph<T>
fn check_stablegraph_memory_usage() {
    use petgraph::stable_graph::StableGraph;
    struct Data {
        id: u64,
    }

    let mut graph: StableGraph<Data, usize> = StableGraph::new();
    let insertion_num = 3_000_000;
    for _ in 0..insertion_num {
        graph.add_node(Data {
            id: Default::default(),
        });
    }
}

#[test]
fn memory_align_of_enum() {
    struct Inner {
        d: u64,
        d2: u32,
        d3: u128,
    }

    struct Data {
        seq_num: u16,
        inner: Inner,
    }

    // actully, the enum is equal with...
    /// struct E {
    ///     tag: MyEnumDiscriminant, // size of one byte
    ///     payload: Data,
    /// }
    /// For a struct, the align padding between tag and payload is the align of payload sub
    /// one byte. Do not think we need to align with the whole size of payload, this a mistake.
    enum E {
        Empty,
        Occupied(Data),
    }

    println!(
        "size of Data memory layout: {}",
        std::mem::size_of::<Data>()
    );
    println!(
        "size of Option<Data> memory layout: {}",
        std::mem::size_of::<Option<Data>>()
    );
    println!("size of E memory layout: {}", std::mem::size_of::<E>());
    println!(
        "size of [Data; 10] memory layout: {}",
        std::mem::size_of::<[Data; 10]>()
    );
    println!(
        "size of [Option<Data>; 10] memory layout: {}",
        std::mem::size_of::<[Option<Data>; 10]>()
    );
    println!(
        "size of usize memory layout: {}",
        std::mem::size_of::<usize>()
    );
    println!(
        "size of Option<isize> memory layout: {}",
        std::mem::size_of::<Option<isize>>()
    );
    println!(
        "size of Option<usize> memory layout: {}",
        std::mem::size_of::<Option<usize>>()
    );

    println!(
        "size of [MaybeUninit<usize>; 10] memory layout: {}",
        std::mem::size_of::<[MaybeUninit<usize>; 10]>()
    );
    println!(
        "size of [Option<usize>; 10] memory layout: {}",
        std::mem::size_of::<[Option<usize>; 10]>()
    );
}

#[test]
fn graph_basic_usage() {
    use petgraph::dot::{Config, Dot};
    use petgraph::stable_graph::StableGraph;
    use std::fs::OpenOptions;

    let mut deps = StableGraph::<&str, &str>::new();
    let pg = deps.add_node("petgraph");
    let fb = deps.add_node("fixedbitset");
    let qc = deps.add_node("quickcheck");
    let rand = deps.add_node("rand");
    let libc = deps.add_node("libc");

    deps.extend_with_edges(&[(pg, fb), (pg, qc), (qc, rand), (rand, libc), (qc, libc)]);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("file/graph.dot");
    if let Ok(mut file) = file {
        let output = Dot::with_config(&deps, &[Config::EdgeNoLabel]);
        file.write_fmt(format_args!("{:?}", output)).unwrap();
    };
}

#[allow(dead_code)]
pub fn check_btree_memory() {
    use btreemap_cus::BTreeMap;

    let mut btree = BTreeMap::new();

    // for (key, val) in key_val_pair.clone().into_iter() {
    //     btree.insert(key, val);
    // }

    let num_of_ele = 1000000;
    let half_of_total = num_of_ele / 2;

    for i in (0..half_of_total).rev() {
        btree.insert(i, format!("aaaaaaaa{i}"));
    }
    for i in half_of_total..num_of_ele {
        btree.insert(i, format!("aaaaaaaa{i}"));
    }

    // for k in 0..num_of_ele {
    //     assert_eq!(
    //         btree.remove(&k),
    //         Some(format!("aaaaaaaa{k}")),
    //         "the value removed is not same with insertion"
    //     );
    // }
    for _ in btree {}
}

pub struct D {
    c: C,
}

pub struct C {}

impl Drop for C {
    fn drop(&mut self) {
        println!("C dropped once");
    }
}

pub fn move_a_struct(p: D) -> D {
    let c = p.c;
    D { c }
}

#[allow(dead_code)]
pub fn vec_usage() {
    let mut v = Vec::new();

    v.push(1);
    v.push(3);
    v.push(2);

    for i in v.iter() {
        println!("{i}");
    }

    println!("after pop the last value");
    v.pop();

    for i in v.iter() {
        println!("{i}");
    }
}

#[allow(dead_code)]
pub fn vec_slice() {
    let v = vec![3, 2, 1];
    assert_eq!(v.as_slice(), &[3, 2, 1]);
}

#[allow(dead_code)]
pub fn vec_basic() {
    let mut v = Vec::new();
    v.push(1);
    // v.into_iter();

    v.iter();
}

#[allow(dead_code)]
pub fn vec_deque_usage() {
    let mut v = VecDeque::new();
    v.push_back(1);
    v.push_front(2);
    v.pop_front();
    v.pop_back();
}

#[allow(dead_code)]
pub fn linked_list() {
    let mut l = LinkedList::new();
    l.push_back(1);
}

#[allow(dead_code)]
pub fn hash_map() {
    let mut hm = HashMap::new();
    hm.insert(1, "hello");
    hm.insert(2, "world");
    hm.insert(3, "!");
}

#[allow(dead_code)]
pub fn btree_map() {
    let mut bm = BTreeMap::new();
    bm.insert(1, "hello");
    bm.insert(2, "world");
    bm.insert(3, "!");
}

#[repr(C)]
struct FieldStruct {
    first: u8,
    second: u16,
    third: u8,
    fourth: u32,
}

#[test]
fn test_mem_align() {
    println!(
        "the align of FieldStruct is: {}",
        std::mem::align_of::<FieldStruct>()
    );
    println!(
        "the size of FieldStruct is: {}",
        std::mem::size_of::<FieldStruct>()
    );

    println!(
        "the align of (u64, FieldStruct) is: {}",
        std::mem::align_of::<(u64, FieldStruct)>()
    );
    println!(
        "the size of (u64, FieldStruct)  is: {}",
        std::mem::size_of::<(u64, FieldStruct)>()
    );
}

#[test]
fn test_modulo_res() {
    const WIDTH: usize = 6;
    let idx = 0usize;
    let bucket_mask = 32usize;
    let idx2 = ((idx.wrapping_sub(WIDTH)) & bucket_mask - 1) + WIDTH;
    println!("the id2 of {idx} is: {idx2}");
    println!(
        "temp value: {}",
        (idx.wrapping_sub(WIDTH)) & bucket_mask - 1
    );
}

#[test]
fn test_slice_api() {
    let slice = [2, 3, 4, 5];
    unsafe {
        println!("the result of unchecked: {:?}", slice.get_unchecked(1..3));
    }
}

#[ignore]
#[test]
fn test_fs_api() -> Result<(), Error> {
    let mut fs = File::open("file/tmp")?;
    let mut buffer: [u8; 10] = [0; 10];
    let len = fs.seek(std::io::SeekFrom::End(0))?;

    println!("the len is {}", len);

    fs.seek(std::io::SeekFrom::Start(1));
    fs.read_exact(&mut buffer);

    println!("{:?}", buffer);

    Ok(())
}

#[test]
fn test_ptr_read_api() {
    struct T {
        n: u8,
    }
    let s = T { n: 11 };
    let s_of_read = unsafe { std::ptr::read(&s) };

    drop(s);

    println!("the value of t: {} ", s_of_read.n);
}

#[test]
fn check_memory_layout() {
    struct K {
        a: [Option<u32>; 30],
    }

    struct M {
        a: [MaybeUninit<u32>; 30],
    }

    // println!("the size of T is: {}", std::mem::size_of::<T>());
    println!("the size of K is: {}", std::mem::size_of::<K>());
    println!("the size of M is: {}", std::mem::size_of::<M>());
}

#[test]
fn mayeb_uninit_usage() {
    unsafe {
        // Tell the compiler that we're initializing a union to an empty case
        let mut results = MaybeUninit::<[bool; 16]>::uninit();

        // VERY CAREFULLY: initialize all the memory
        // DO NOT: create an &mut [bool; 16]
        // DO NOT: create an &mut bool
        let arr_ptr = results.as_mut_ptr() as *mut bool;
        for i in 0..16 {
            arr_ptr.add(i).write(i % 2 == 0);
        }

        // All values carefully proven by programmer to be init
        let results_ref = &*results.as_ptr();
        println!("{:?}", results_ref);
    }
}

#[test]
fn maybe_uninit_usage2() {
    #[derive(Debug, PartialEq)]
    pub struct Foo {
        name: String,
        list: Vec<u8>,
    }

    let foo = {
        let mut uninit: MaybeUninit<Foo> = MaybeUninit::uninit();
        let ptr = uninit.as_mut_ptr();

        // Initializing the `name` field
        // Using `write` instead of assignment via `=` to not call `drop` on the
        // old, uninitialized value.
        unsafe {
            addr_of_mut!((*ptr).name).write("Bob".to_string());
        }

        // Initializing the `list` field
        // If there is a panic here, then the `String` in the `name` field leaks.
        unsafe {
            addr_of_mut!((*ptr).list).write(vec![0, 1, 2]);
        }

        // All the fields are initialized, so we call `assume_init` to get an initialized Foo.
        unsafe { uninit.assume_init() }
    };

    assert_eq!(
        foo,
        Foo {
            name: "Bob".to_string(),
            list: vec![0, 1, 2]
        }
    );
}

#[test]
fn maybe_uninit_usage3() {
    let data = {
        // Create an uninitialized array of `MaybeUninit`. The `assume_init` is
        // safe because the type we are claiming to have initialized here is a
        // bunch of `MaybeUninit`s, which do not require initialization.
        let mut data: [MaybeUninit<Vec<u32>>; 1000] = unsafe {
            let t = MaybeUninit::uninit();
            t.assume_init()
        };

        // Dropping a `MaybeUninit` does nothing, so if there is a panic during this loop,
        // we have a memory leak, but there is no memory safety issue.
        for elem in &mut data[..] {
            elem.write(vec![42]);
        }

        // Everything is initialized. Transmute the array to the
        // initialized type.
        // unsafe { std::mem::transmute::<_, [Vec<u32>; 1000]>(data) }
        unsafe { MaybeUninit::array_assume_init(data) }
    };

    assert_eq!(&data[0], &[42]);
}

// fn silly_default<const N: usize, T: Sized>(arr: [MaybeUninit<T>; N]) -> [T; N] {
//     unsafe { core::mem::transmute::<_, _>(arr) }
// }

#[test]
fn test_fn_once() {
    fn call_with_cb<F: FnOnce()>(f: F) {
        f();
    }
    let mut flag = false;
    call_with_cb(|| {
        flag = true;
    });

    println!("the val of flag: {}", flag);
}

#[test]
fn test_assign() {
    struct A {
        v: Option<[u8; 10]>,
    }

    struct B {
        v: [u8; 10],
    }

    let mut a = A { v: Some([8; 10]) };
    // warining: this will not modify the value inside of v, because we use `unwrap()`, which move
    // the v to another place. And if the Type Option contains is Copy, the v is Copy too, we won't
    // get error hint from compilier.
    a.v.unwrap()[0] = 10;

    let mut b = B { v: [8; 10] };
    b.v[0] = 10;

    println!("the value of a: {:?}", a.v.unwrap());
    println!("the value of b: {:?}", b.v);

    // Note: the associated func `unwrap` will move the x, if x contains a Copy type, which
    // we could easyly ignore this kind of mistake.
    // impl<T> Copy for Option<T>
    // where
    //     T: Copy,
    let x = Some("data");
    println!("{:?}", x.unwrap());
    println!("{:?}", x.unwrap());
}

#[test]
fn check_ptr_addr_of_world() {
    #[allow(dead_code)]
    #[derive(Debug)]
    struct T {
        a: usize,
    }
    let a = 1;
    let b = 1;
    let c = T { a: 0 };
    let p_c = &c;
    let pp_c = &p_c;
    println!("the addr of a is: {:?}", std::ptr::addr_of!(a));
    println!("the addr of b is: {:?}", std::ptr::addr_of!(b));
    println!("the addr of c is: {:?}", std::ptr::addr_of!(c));
    println!("the addr of c is: {:p}", p_c);
    println!("the addr of pointer of c is: {:p}", pp_c);
}

#[test]
fn variable_shadow_test() {
    #[derive(Debug)]
    struct Dummy(usize);
    impl Drop for Dummy {
        fn drop(&mut self) {
            println!("{:?} is droped, pointer location: {:p}", *self, self);
        }
    }

    let mut d = Dummy(3);

    {
        let mut closure = || {
            // 已有变量被重新赋值时只是对引用指向的内存区域的内容的覆盖，而之前的
            // 内存区域的对象被 Drop了，这并不会改变已有变量的所有权，可以简单的认为赋值操作是
            // std::ptr::drop_in_place(&mut d); 这个和 drop(std::ptr::read(&d)) 是差不多的
            // 接着，
            // std::ptr::write(&mut d, NewValue)
            // 这个过程中并未发生 d 的 move，只需要 borrow mut d 即可。
            // 只有当 d 当作表达式赋值给其它变量的时候，d 才会发生 move，而被赋值则不属于这种情况
            d = Dummy(5);
        };

        closure();

        // let d = Dummy(7);
        // drop(d);
    }

    println!("dummy = {:?}, pointer location: {:p}", d, &d);
}
