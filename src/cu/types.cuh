#pragma once

namespace gplda {

typedef float f32;
typedef int i32;
typedef unsigned int u32;
typedef unsigned long long int u64;

static_assert(sizeof(u64) == 8, "u64 is not 8 bytes");
static_assert(sizeof(u32) == 4, "u32 is not 4 bytes");
static_assert(sizeof(i32) == 4, "i32 is not 4 bytes");
static_assert(sizeof(f32) == 4, "f32 is not 4 bytes");

enum SynchronizationType {warp, block};

}
