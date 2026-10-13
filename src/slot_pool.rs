pub struct SlotPool<T> {
    slots: Vec<Option<T>>,
    frees: Vec<SlotPoolHandle>,
}

impl<T: Debug> Debug for SlotPool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.iter().for_each(|value| {
            writeln!(f, "{:?}", value).unwrap();
        });
        Ok(())
    }
}

impl<T> Default for SlotPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

mod handle {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct SlotPoolHandle(u32);

    impl SlotPoolHandle {
        pub const NULL: Self = Self(u32::MAX);

        #[inline]
        pub(super) const fn from_usize(index: usize) -> Option<Self> {
            if index >= u32::MAX as usize {
                None
            } else {
                Some(Self(index as u32))
            }
        }

        #[inline]
        pub(super) const fn as_usize(&self) -> usize {
            self.0 as usize
        }
    }
}

use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

pub use handle::SlotPoolHandle;

impl<T> SlotPool<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            frees: Vec::new(),
        }
    }

    #[inline]
    pub fn insert(&mut self, value: T) -> SlotPoolHandle {
        if let Some(free) = self.frees.pop() {
            self.slots[free.as_usize()] = Some(value);

            free
        } else {
            let index = self.slots.len();
            self.slots.push(Some(value));

            SlotPoolHandle::from_usize(index).unwrap()
        }
    }

    #[inline]
    pub fn allocate_slot(&mut self) -> SlotPoolHandle {
        if let Some(free) = self.frees.pop() {
            free
        } else {
            let index = self.slots.len();
            self.slots.push(None);

            SlotPoolHandle::from_usize(index).unwrap()
        }
    }

    #[inline]
    pub fn insert_at(&mut self, value: T, handle: SlotPoolHandle) -> Result<(), T> {
        if let Some(slot) = self.slots.get_mut(handle.as_usize()) {
            if slot.is_none() {
                *slot = Some(value);
                Ok(())
            } else {
                Err(value)
            }
        } else {
            Err(value)
        }
    }

    #[inline]
    pub fn get(&self, handle: SlotPoolHandle) -> Option<&T> {
        self.slots.get(handle.as_usize())?.as_ref()
    }

    #[cfg(not(debug_assertions))]
    #[inline]
    pub unsafe fn get_unchecked(&self, handle: SlotPoolHandle) -> &T {
        unsafe {
            self.slots
                .get_unchecked(handle.as_usize())
                .as_ref()
                .unwrap_unchecked()
        }
    }

    #[inline]
    pub fn get_mut(&mut self, handle: SlotPoolHandle) -> Option<&mut T> {
        self.slots.get_mut(handle.as_usize())?.as_mut()
    }

    // #[cfg(not(debug_assertions))]
    // #[inline]
    // pub unsafe fn get_unchecked_mut(&mut self, handle: SlotPoolHandle) -> &mut T {
    //     unsafe {
    //         self.slots
    //             .get_unchecked_mut(handle.as_usize())
    //             .as_mut()
    //             .unwrap_unchecked()
    //     }
    // }

    #[inline]
    pub fn remove(&mut self, handle: SlotPoolHandle) -> Option<T> {
        if let Some(slot) = self.slots.get_mut(handle.as_usize()) {
            if let Some(value) = slot.take() {
                self.frees.push(handle);
                Some(value)
            } else {
                None
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(Option::as_ref)
    }
}

impl<T> Index<SlotPoolHandle> for SlotPool<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: SlotPoolHandle) -> &Self::Output {
        self.slots[index.as_usize()].as_ref().unwrap()
    }
}

impl<T> IndexMut<SlotPoolHandle> for SlotPool<T> {
    #[inline]
    fn index_mut(&mut self, index: SlotPoolHandle) -> &mut Self::Output {
        self.slots[index.as_usize()].as_mut().unwrap()
    }
}
