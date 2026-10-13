use std::num::NonZeroU32;

pub struct GenSlotPool<T> {
    slots: Vec<GenSlot<T>>,
    frees: Vec<u32>,
}

/// Using enum here because rust can optimize its size.
/// For example `struct TestStruct(GenSlot<u32>)` is 8 bytes.
enum GenSlot<T> {
    Occupied { generation: NonZeroU32, data: T },
    Empty { generation: NonZeroU32 },
}

mod handle {
    use std::num::NonZeroU32;

    pub struct GenSlotPoolHandle {
        index: u32,
        generation: NonZeroU32,
    }

    impl GenSlotPoolHandle {
        pub const NULL: Self = Self {
            index: u32::MAX,
            generation: NonZeroU32::new(1).unwrap(),
        };

        #[inline]
        pub(super) const fn from_size(index: usize, generation: NonZeroU32) -> Option<Self> {
            if index >= u32::MAX as usize {
                None
            } else {
                Some(Self {
                    index: index as u32,
                    generation,
                })
            }
        }

        #[inline]
        pub(super) const fn as_usize(&self) -> usize {
            self.index as usize
        }
    }
}
pub use handle::GenSlotPoolHandle;

impl<T> GenSlotPool<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            frees: Vec::new(),
        }
    }
}
