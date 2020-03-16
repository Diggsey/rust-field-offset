// Explicit lifetimes are clearer when we are working with raw pointers,
// as the compiler will not warn us if we specify lifetime constraints
// which are too lax.
#![allow(clippy::needless_lifetimes)]

use std::marker::PhantomData;
use std::mem;
use std::ops::Add;
use std::fmt;

#[doc(hidden)]
pub extern crate memoffset as __memoffset; // `pub` for macro availability

/// Represents a pointer to a field of type `U` within the type `T`
#[repr(transparent)]
pub struct FieldOffset<T, U>(
    /// Offset in bytes of the field within the struct
    usize,
    /// A pointer-to-member can be thought of as a function from
    /// `&T` to `&U` with matching lifetimes
    PhantomData<dyn for<'a> Fn(&'a T) -> &'a U>
);

impl<T, U> FieldOffset<T, U> {
    // Use MaybeUninit to get a fake T
    #[cfg(fieldoffset_maybe_uninit)]
    #[inline]
    fn with_uninit_ptr<R, F: FnOnce(*const T) -> R>(f: F) -> R {
        let uninit = mem::MaybeUninit::<T>::uninit();
        f(uninit.as_ptr())
    }

    // Use a dangling pointer to get a fake T
    #[cfg(not(fieldoffset_maybe_uninit))]
    #[inline]
    fn with_uninit_ptr<R, F: FnOnce(*const T) -> R>(f: F) -> R {
        f(mem::align_of::<T>() as *const T)
    }

    /// Construct a field offset via a lambda which returns a reference
    /// to the field in question.
    ///
    /// # Safety
    /// 
    /// The lambda *must not* dereference the provided pointer or access the
    /// inner value in any way as it may point to uninitialized memory.
    /// 
    /// For the returned `FieldOffset` to be safe to use, the returned pointer
    /// must be valid for *any* instance of `T`. For example, returning a pointer
    /// to a field from an enum with multiple variants will produce a `FieldOffset`
    /// which is unsafe to use.
    pub unsafe fn new<F: for<'a> FnOnce(*const T) -> *const U>(f: F) -> Self {
        let offset = Self::with_uninit_ptr(|base_ptr| {
            let field_ptr = f(base_ptr);
            (field_ptr as usize).wrapping_sub(base_ptr as usize)
        });

        // Construct an instance using the offset
        Self::new_from_offset(offset)
    }
    /// Construct a field offset directly from a byte offset.
    /// 
    /// # Safety
    /// 
    /// For the returned `FieldOffset` to be safe to use, the field offset
    /// must be valid for *any* instance of `T`. For example, returning the offset
    /// to a field from an enum with multiple variants will produce a `FieldOffset`
    /// which is unsafe to use.
    #[inline]
    pub unsafe fn new_from_offset(offset: usize) -> Self {
        // Sanity check: ensure that the field offset plus the field size
        // is no greater than the size of the containing struct. This is
        // not sufficient to make the function *safe*, but it does catch
        // obvious errors like returning a reference to a boxed value,
        // which is owned by `T` and so has the correct lifetime, but is not
        // actually a field.
        assert!(offset + mem::size_of::<U>() <= mem::size_of::<T>());

        FieldOffset(offset, PhantomData)
    }

    // Methods for applying the pointer to member

    /// Apply the field offset to a native pointer.
    #[inline]
    pub fn apply_ptr(self, x: *const T) -> *const U {
        ((x as usize) + self.0) as *const U
    }
    /// Apply the field offset to a native mutable pointer.
    #[inline]
    pub fn apply_ptr_mut(self, x: *mut T) -> *mut U {
        ((x as usize) + self.0) as *mut U
    }
    /// Apply the field offset to a reference.
    #[inline]
    pub fn apply<'a>(self, x: &'a T) -> &'a U {
        unsafe { &*self.apply_ptr(x) }
    }
    /// Apply the field offset to a mutable reference.
    #[inline]
    pub fn apply_mut<'a>(self, x: &'a mut T) -> &'a mut U {
        unsafe { &mut *self.apply_ptr_mut(x) }
    }
    /// Get the raw byte offset for this field offset.
    #[inline]
    pub fn get_byte_offset(self) -> usize {
        self.0
    }

    // Methods for unapplying the pointer to member

    /// Unapply the field offset to a native pointer.
    /// 
    /// # Safety
    /// 
    /// *Warning: very unsafe!*
    /// 
    /// This applies a negative offset to a pointer. If the safety
    /// implications of this are not already clear to you, then *do
    /// not* use this method. Also be aware that Rust has stronger
    /// aliasing rules than other languages, so it may be UB to
    /// dereference the resulting pointer even if it points to a valid
    /// location, due to the presence of other live references.
    #[inline]
    pub unsafe fn unapply_ptr(self, x: *const U) -> *const T {
        ((x as usize) - self.0) as *const T
    }
    /// Unapply the field offset to a native mutable pointer.
    ///
    /// # Safety
    /// 
    /// *Warning: very unsafe!*
    /// 
    /// This applies a negative offset to a pointer. If the safety
    /// implications of this are not already clear to you, then *do
    /// not* use this method. Also be aware that Rust has stronger
    /// aliasing rules than other languages, so it may be UB to
    /// dereference the resulting pointer even if it points to a valid
    /// location, due to the presence of other live references.
    #[inline]
    pub unsafe fn unapply_ptr_mut(self, x: *mut U) -> *mut T {
        ((x as usize) - self.0) as *mut T
    }
    /// Unapply the field offset to a reference.
    ///
    /// # Safety
    /// 
    /// *Warning: very unsafe!*
    /// 
    /// This applies a negative offset to a reference. If the safety
    /// implications of this are not already clear to you, then *do
    /// not* use this method. Also be aware that Rust has stronger
    /// aliasing rules than other languages, so this method may cause UB
    /// even if the resulting reference points to a valid location, due
    /// to the presence of other live references.
    #[inline]
    pub unsafe fn unapply<'a>(self, x: &'a U) -> &'a T {
        &*self.unapply_ptr(x)
    }
    /// Unapply the field offset to a mutable reference.
    ///
    /// # Safety
    /// 
    /// *Warning: very unsafe!*
    /// 
    /// This applies a negative offset to a reference. If the safety
    /// implications of this are not already clear to you, then *do
    /// not* use this method. Also be aware that Rust has stronger
    /// aliasing rules than other languages, so this method may cause UB
    /// even if the resulting reference points to a valid location, due
    /// to the presence of other live references.
    #[inline]
    pub unsafe fn unapply_mut<'a>(self, x: &'a mut U) -> &'a mut T {
        &mut *self.unapply_ptr_mut(x)
    }
}

/// Allow chaining pointer-to-members.
///
/// Applying the resulting field offset is equivalent to applying the first
/// field offset, then applying the second field offset.
///
/// The requirements on the generic type parameters ensure this is a safe operation.
impl<T, U, V> Add<FieldOffset<U, V>> for FieldOffset<T, U> {
    type Output = FieldOffset<T, V>;

    #[inline]
    fn add(self, other: FieldOffset<U, V>) -> FieldOffset<T, V> {
        FieldOffset(self.0 + other.0, PhantomData)
    }
}

/// The debug implementation prints the byte offset of the field in hexadecimal.
impl<T, U> fmt::Debug for FieldOffset<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "FieldOffset({:#x})", self.0)
    }
}

impl<T, U> Copy for FieldOffset<T, U> { }
impl<T, U> Clone for FieldOffset<T, U> {
    fn clone(&self) -> Self { *self }
}

/// This macro allows safe construction of a FieldOffset,
/// by generating a known to be valid lambda to pass to the
/// constructor. It takes a type and the identifier of a field
/// within that type as input.
///
/// Examples:
///
/// Offset of field `Foo.bar`
///
/// ```rust
/// # #[macro_use]
/// # extern crate field_offset;
/// # fn main() {
/// #[repr(C)]
/// struct Foo { foo: i32, bar: i32 }
/// assert_eq!(offset_of!(Foo => bar).get_byte_offset(), 4);
/// # }
/// ```
///
/// Offset of nested field `Foo.bar.x`
///
/// ```rust
/// # #[macro_use]
/// # extern crate field_offset;
/// # fn main() {
/// struct Bar { a: u8, x: u8 }
/// struct Foo { foo: i32, bar: Bar }
/// assert_eq!(offset_of!(Foo => bar: Bar => x).get_byte_offset(), 5);
/// # }
/// ```
#[macro_export]
macro_rules! offset_of {
    ($t: tt => $f: tt) => {{
        // Construct the offset
        #[allow(unused_unsafe)]
        unsafe {
            $crate::FieldOffset::<$t, _>::new(|x| {
                $crate::__memoffset::raw_field!(x, $t, $f)
            })
        }
    }};
    ($t: path => $f: ident: $($rest: tt)*) => {
        offset_of!($t => $f) + offset_of!($($rest)*)
    };
}

#[cfg(test)]
mod tests {
    // Example structs
    #[derive(Debug)]
    struct Foo {
        a: u32,
        b: f64,
        c: bool
    }

    #[derive(Debug)]
    struct Bar {
        x: u32,
        y: Foo,
    }

    #[derive(Debug)]
    struct Tuple(i32, f64);

    #[test]
    fn test_simple() {
        // Get a pointer to `b` within `Foo`
        let foo_b = offset_of!(Foo => b);

        // Construct an example `Foo`
        let mut x = Foo {
            a: 1,
            b: 2.0,
            c: false
        };

        // Apply the pointer to get at `b` and read it
        {
            let y = foo_b.apply(&x);
            assert!(*y == 2.0);
        }

        // Apply the pointer to get at `b` and mutate it
        {
            let y = foo_b.apply_mut(&mut x);
            *y = 42.0;
        }
        assert!(x.b == 42.0);
    }

    #[test]
    fn test_tuple() {
        // Get a pointer to `b` within `Foo`
        let tuple_1 = offset_of!(Tuple => 1);

        // Construct an example `Foo`
        let mut x = Tuple(1, 42.0);

        // Apply the pointer to get at `b` and read it
        {
            let y = tuple_1.apply(&x);
            assert!(*y == 42.0);
        }

        // Apply the pointer to get at `b` and mutate it
        {
            let y = tuple_1.apply_mut(&mut x);
            *y = 5.0;
        }
        assert!(x.1 == 5.0);
    }

    #[test]
    fn test_nested() {
        // Construct an example `Foo`
        let mut x = Bar {
            x: 0,
            y: Foo {
                a: 1,
                b: 2.0,
                c: false
            }
        };

        // Combine the pointer-to-members
        let bar_y_b = offset_of!(Bar => y: Foo => b);

        // Apply the pointer to get at `b` and mutate it
        {
            let y = bar_y_b.apply_mut(&mut x);
            *y = 42.0;
        }
        assert!(x.y.b == 42.0);
    }
}
