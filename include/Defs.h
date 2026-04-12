#pragma once

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#define RESTRICT __restrict
#define PRAGMA_IVDEP __pragma(loop(ivdep))
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define PRAGMA_IVDEP _Pragma("GCC ivdep")
#endif

#define DISALLOW_COPY_AND_ASSIGN(Type)\
	Type(const Type&) = delete;\
	Type& operator=(const Type&) = delete

#define DISALLOW_COPY_MOVE_AND_ASSIGN(Type)\
	Type(const Type&) = delete;\
	Type& operator=(const Type&) = delete;\
	Type(Type&&) = delete;\
	Type& operator=(Type&&) = delete

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(no_unique_address) >= 201803L
#define NO_UNIQUE_ADDRESS [[no_unique_address]]
#elif __has_cpp_attribute(msvc::no_unique_address)
#define NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define NO_UNIQUE_ADDRESS
#endif
#else
#define NO_UNIQUE_ADDRESS
#endif
