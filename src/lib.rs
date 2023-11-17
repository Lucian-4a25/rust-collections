#![feature(dropck_eyepatch)]
#![feature(exclusive_range_pattern)]
#![feature(const_trait_impl)]
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(strict_provenance)]
#![feature(associated_type_defaults)]
#![feature(linked_list_remove)]

extern crate alloc;

pub mod binary_heap_cus;
pub mod btreemap_cus;
pub mod graph;
pub mod linked_list_cus;
pub mod lru;
mod raw_vec;
pub mod vec_cus;
pub mod vecdeque_cus;
