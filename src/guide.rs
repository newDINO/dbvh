//! # User guide
//! This crate supports generic bounding volume using traits:
//!
//! - [`BoundingVolume`](crate::BoundingVolume) is the basic trait required to build a [`Bvh`](crate::Bvh).
//! - [`IntersectSelf`](crate::IntersectSelf) need to be implemented for finding possible intersections using [`Bvh`](crate::Bvh).
//! - [`RayCast`](crate::RayCast) need to be implemented for using [`Bvh`](crate::Bvh) to accelarate ray casting.
//! - [`Enlarged`](crate::Enlarge) need to be implemented for using [`EnlargedBvh`](crate::EnlargedBvh).
//!
//!   It is recommended to use [`EnlargedBvh`](crate::EnlargedBvh) for dynamic scenes,
//!   because leaves of the moving objects will only be re-inserted if it moves out of the enlarged bounding volume.
//!
//! Below are 2 example implementations of these traits.
//!
//! # Example AABB implementation
//! This is a 3D AABB implementation of all the traits listed above,
//! based on [`nalgebra`](https://docs.rs/nalgebra/latest/nalgebra/).
//! ```
#![doc = include_str!("../common/aabb.rs")]
//! ```
//!
//! # Example bounding sphere implementation
//! This is a 3D bounding sphere implementation of all the traits listed above,
//! based on [`glam`](https://docs.rs/glam/latest/glam/).
//! ```
#![doc = include_str!("../common/sphere.rs")]
//! ```
