#pragma once
#include "Defs.h"
#include <cassert>

#ifdef _WIN32
#include <malloc.h>
#endif

// Platform-specific aligned allocation helpers
FORCE_INLINE void* AlignedAlloc(size_t size, size_t alignment)
{
	assert(size);
	assert((alignment & (alignment - 1)) == 0);

#ifdef _WIN32
	return _aligned_malloc(size, alignment);
#else
	void* ptr = nullptr;
	size_t adjAlignment = (alignment < sizeof(void*)) ? sizeof(void*) : alignment;
	if (posix_memalign(&ptr, adjAlignment, size) != 0)
	{
		return nullptr;
	}
	return ptr;
#endif
}

FORCE_INLINE void AlignedFree(void* ptr)
{
	assert(ptr);
#ifdef _WIN32
	_aligned_free(ptr);
#else
	free(ptr);
#endif
}
