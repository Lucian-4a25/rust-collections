use std::{alloc::Layout, ptr::NonNull};

pub struct RawVec<T> {
    ptr: NonNull<T>,
    cap: usize,
}

impl<T> RawVec<T> {
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            cap: if std::mem::size_of::<T>() == 0 {
                usize::MAX
            } else {
                0
            },
        }
    }

    #[inline]
    pub fn cap(&self) -> usize {
        self.cap
    }

    #[inline]
    pub fn ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn grow(&mut self) {
        // since we set the capacity to usize::MAX when T has size 0,
        // getting to here necessarily means the Vec is overfull.
        assert!(std::mem::size_of::<T>() != 0, "capacity overflow");

        let (new_cap, new_layout) = if self.cap == 0 {
            (1, Layout::array::<T>(1).unwrap())
        } else {
            // this is not possible to overflow cause self.cap <= isize.MAX
            let new_cap = self.cap * 2;

            (new_cap, Layout::array::<T>(new_cap).unwrap())
        };

        // Ensure that the new allocation doesn't exceed `isize::MAX` bytes.
        assert!(
            new_layout.size() <= isize::MAX as usize,
            "Allocation too large"
        );

        let new_ptr = if self.cap == 0 {
            unsafe { alloc::alloc::alloc(new_layout) }
        } else {
            let old_layout = Layout::array::<T>(self.cap).unwrap();
            let old_ptr = self.ptr.as_ptr() as *mut u8;
            unsafe { alloc::alloc::realloc(old_ptr, old_layout, new_layout.size()) }
        };

        // If allocation fails, `new_ptr` will be null, in which case we abort.
        self.ptr = match NonNull::new(new_ptr as *mut T) {
            Some(p) => p,
            None => alloc::alloc::handle_alloc_error(new_layout),
        };
        self.cap = new_cap;
    }

    // pub fn make_contiguous(&mut self, end: usize) {
    //   if self.cap > 0 && self.cap % 2 != 0 {
    //     let start_ptr = self.ptr() as usize + cap / 2;
    //     // unsafe {
    //     //   alloc::alloc::alloc()
    //     // }
    //   }
    // }
}

impl<T> Drop for RawVec<T> {
    fn drop(&mut self) {
        let elem_size = std::mem::size_of::<T>();

        if self.cap != 0 && elem_size != 0 {
            unsafe {
                let layout = Layout::array::<T>(self.cap).unwrap();
                alloc::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

pub struct RawValIter<T> {
    start: *const T,
    end: *const T,
}

impl<T> RawValIter<T> {
    pub unsafe fn new(slice: &[T]) -> Self {
        Self {
            start: slice.as_ptr(),
            end: if std::mem::size_of::<T>() == 0 {
                // 这是一个没有意义的指针，这里对指针只是一个用于遍历计数的作用，因为 pointer offset API 对 ZSTs 类型无效
                (slice.as_ptr() as usize + slice.len()) as *const _
            } else if slice.len() == 0 {
                slice.as_ptr()
            } else {
                slice.as_ptr().add(slice.len())
            },
        }
    }

    pub fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                if std::mem::size_of::<T>() == 0 {
                    self.start = (self.start as usize + 1) as *const _;
                    Some(std::ptr::read(NonNull::dangling().as_ptr()))
                } else {
                    let r = std::ptr::read(self.start);
                    self.start = self.start.add(1);
                    Some(r)
                }
            }
        }
    }

    pub fn size_hint(&self) -> (usize, Option<usize>) {
        let elem_size = std::mem::size_of::<T>();
        let len =
            (self.end as usize - self.start as usize) / if elem_size == 0 { 1 } else { elem_size };
        (len, Some(len))
    }

    pub fn next_back(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                if std::mem::size_of::<T>() == 0 {
                    self.end = (self.end as usize - 1) as *const _;
                    Some(std::ptr::read(NonNull::dangling().as_ptr()))
                } else {
                    self.end = self.end.sub(1);
                    let r = std::ptr::read(self.end);
                    Some(r)
                }
            }
        }
    }
}
