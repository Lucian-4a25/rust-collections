use crate::raw_vec::RawVec;

pub struct VecDeque<T> {
    head: usize,
    len: usize,
    buf: RawVec<T>,
}

#[inline]
fn wrap_index(logical_index: usize, cap: usize) -> usize {
    if logical_index >= cap {
        logical_index - cap
    } else {
        logical_index
    }
}

#[allow(dead_code)]
impl<T> VecDeque<T> {
    pub fn new() -> Self {
        VecDeque {
            head: 0,
            len: 0,
            buf: RawVec::new(),
        }
    }

    fn wrap_sub(&self, offset: usize) -> usize {
        let v = self.head.wrapping_sub(offset).wrapping_add(self.buf.cap());
        wrap_index(v, self.buf.cap())
    }

    fn wrap_add(&self, offset: usize) -> usize {
        wrap_index(self.head.wrapping_add(offset), self.buf.cap())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push_front(&mut self, v: T) {
        if self.len == self.buf.cap() {
            self.grow_ring_buffer();
        }

        unsafe {
            self.head = self.wrap_sub(1);
            self.buf.ptr().add(self.head).write(v);
            self.len += 1;
        }

        // println!("{}", self.head);
    }

    pub fn push_back(&mut self, v: T) {
        if self.len == self.buf.cap() {
            self.grow_ring_buffer();
        }

        unsafe {
            self.buf.ptr().add(self.wrap_add(self.len)).write(v);
            self.len += 1;
        }

        // println!("{}", self.head);
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                let r = self.buf.ptr().add(self.head).read();
                self.head = self.wrap_add(1);
                self.len -= 1;
                Some(r)
            }
        }
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                let r = self.buf.ptr().add(self.wrap_add(self.len)).read();

                Some(r)
            }
        }
    }

    // check if the ring buffer is contiguous
    pub fn is_contiguous(&self) -> bool {
        self.head + self.len <= self.buf.cap()
    }

    pub fn grow_ring_buffer(&mut self) {
        let is_contiguous = self.is_contiguous();
        let old_cap = self.buf.cap();
        let head_len = old_cap - self.head;
        let trail_len = self.len - head_len;
        self.buf.grow();
        // we need to write the startr part of ring to the end part.
        if !is_contiguous {
            unsafe {
                // we could move the less part of the ring buffer
                if head_len > trail_len {
                    std::ptr::copy_nonoverlapping(
                        self.buf.ptr(),
                        self.buf.ptr().add(old_cap),
                        trail_len,
                    );
                    // PS: we could leverage the ptr::copy_nonoverlapping API
                    // for i in 0..self.head {
                    //     let old = self.buf.ptr().add(i).read();
                    //     self.buf
                    //         .ptr()
                    //         .add(self.wrap_add(new_part_offset + i))
                    //         .write(old);
                    // }
                } else {
                    let new_head = self.buf.cap() - head_len;
                    std::ptr::copy_nonoverlapping(
                        self.buf.ptr().add(self.head),
                        self.buf.ptr().add(new_head),
                        head_len,
                    );
                    self.head = new_head;
                }
            }
        }
    }
}

// impl<T> Deref for VecDeque<T> {
//     type Target = &[T];
// }

// impl<T> IntoIterator for VecDeque<T> {
//     type Item = T;
//     fn into_iter(self) -> Self::IntoIter {
//         // let ptr = self.buf.ptr();
//         // RawValIter::new()
//     }
// }

#[cfg(test)]
mod vecdeque_test {
    use super::*;

    #[test]
    fn test_vecdeque_pop_back() {
        let mut v = VecDeque::new();
        // assert_eq!(
        //     0usize.wrapping_sub(1),
        //     usize::MAX,
        //     "the result of wrapping sub is not from 0 to usize::MAX"
        // );

        v.push_back(3);
        v.push_back(4);
        v.push_front(2);
        v.push_front(1);
        v.push_back(5);
        assert_eq!(v.pop_back().unwrap(), 5);
        assert_eq!(v.pop_back().unwrap(), 4);
        assert_eq!(v.pop_back().unwrap(), 3);
        assert_eq!(v.pop_back().unwrap(), 2);
        assert_eq!(v.pop_back().unwrap(), 1);
        assert_eq!(v.pop_back(), None);
    }

    #[test]
    fn test_vecdeque_pop_front() {
        let mut v = VecDeque::new();

        v.push_back(3);
        v.push_back(4);
        v.push_front(2);
        v.push_front(1);
        v.push_back(5);
        assert_eq!(v.pop_front().unwrap(), 1);
        assert_eq!(v.pop_front().unwrap(), 2);
        assert_eq!(v.pop_front().unwrap(), 3);
        assert_eq!(v.pop_front().unwrap(), 4);
        assert_eq!(v.pop_front().unwrap(), 5);
        assert_eq!(v.pop_back(), None);
    }
}
