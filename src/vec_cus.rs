use crate::raw_vec::RawValIter;
use crate::raw_vec::RawVec;
use std::alloc::Layout;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
pub struct Vec<T> {
    pub buf: RawVec<T>,
    len: usize,
}

unsafe impl<T: Send> Send for Vec<T> {}
unsafe impl<T: Sync> Sync for Vec<T> {}

#[allow(dead_code)]
impl<T> Vec<T> {
    pub fn new() -> Self {
        assert_ne!(
            std::mem::align_of::<T>(),
            0,
            "we're not going to hand ZSTs."
        );
        Self {
            buf: RawVec::new(),
            len: 0,
        }
    }

    pub fn push(&mut self, elem: T) {
        if self.len == self.buf.cap() {
            self.buf.grow();
        };

        unsafe {
            std::ptr::write(self.buf.ptr().add(self.len), elem);
            self.len += 1;
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            let r = unsafe { std::ptr::read(self.buf.ptr().add(self.len)) };
            Some(r)
        }
    }

    pub fn insert(&mut self, index: usize, elem: T) {
        assert!(index <= self.len, "index out of bounds");
        if self.buf.cap() == self.len {
            self.buf.grow();
        }

        unsafe {
            // ptr::copy(src, dest, len): "copy from src to dest len elems"
            std::ptr::copy(
                self.buf.ptr().add(index),
                self.buf.ptr().add(index + 1),
                self.len - index,
            );
            std::ptr::write(self.buf.ptr().add(index), elem);
            self.len += 1;
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        // Note: `<` because it's *not* valid to remove after everything
        assert!(index < self.len, "index out of bounds");
        unsafe {
            self.len -= 1;
            let result = std::ptr::read(self.buf.ptr().add(index));
            std::ptr::copy(
                self.buf.ptr().add(index + 1),
                self.buf.ptr().add(index),
                self.len - index,
            );
            result
        }
    }

    pub fn drain(&mut self /* range: [usize; 2] */) -> Drain<'_, T> {
        unsafe {
            let iter = RawValIter::new(self);
            // this is a mem::forget safety thing. If Drain is forgotten, we just
            // leak the whole Vec's contents. Also we need to do this *eventually*
            // anyway, so why not do it now?
            self.len = 0;

            Drain {
                iter,
                vec: PhantomData,
            }
        }
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        if self.buf.cap() != 0 {
            while let Some(_) = self.pop() {}
        }
    }
}

impl<T> Deref for Vec<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.buf.ptr(), self.len) }
    }
}

impl<T> DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buf.ptr(), self.len) }
    }
}

pub struct IntoIter<T> {
    _buf: RawVec<T>,
    iter: RawValIter<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<T> IntoIterator for Vec<T> {
    type IntoIter = IntoIter<T>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let iter = RawValIter::new(&self);
            // Make sure not to drop Vec since that would free the buffer
            let buf = std::ptr::read(&self.buf);

            std::mem::forget(self);

            IntoIter { iter, _buf: buf }
        }
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        if self._buf.cap() != 0 {
            for _ in &mut *self {}
            let layout = Layout::array::<T>(self._buf.cap()).unwrap();
            unsafe {
                alloc::alloc::dealloc(self._buf.ptr() as *mut u8, layout);
            }
        }
    }
}

pub struct Drain<'a, T: 'a> {
    vec: PhantomData<&'a mut Vec<T>>,
    iter: RawValIter<T>,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

#[cfg(test)]
mod vec_api_test {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut v = Vec::new();
        v.push(1);
        v.push(2);
        v.push(3);
        let mut iter = v.iter();
        assert_eq!(1, *iter.next().unwrap());
        assert_eq!(2, *iter.next().unwrap());
        assert_eq!(3, *iter.next().unwrap());

        assert_eq!(3, v.pop().unwrap());
        assert_eq!(2, v.pop().unwrap());
        assert_eq!(1, v.pop().unwrap());
    }

    #[test]
    fn test_insert() {
        let mut v = Vec::new();
        v.push(1);
        v.push(3);
        v.insert(1, 2);

        assert_eq!(3, v.pop().unwrap());
        assert_eq!(2, v.pop().unwrap());
        assert_eq!(1, v.pop().unwrap());
    }

    #[test]
    fn test_remove() {
        let mut v = Vec::new();
        v.push(1);
        v.push(3);
        v.remove(0);

        assert_eq!(3, v.pop().unwrap());
    }
}
