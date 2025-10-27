
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

#define DECLARE_SQ4_FUNCTIONS(ns)                           \
    namespace ns {                                          \
    float                                                   \
    SQ4ComputeIP(const float* RESTRICT query,               \
                 const uint8_t* RESTRICT codes,             \
                 const float* RESTRICT lower_bound,         \
                 const float* RESTRICT diff,                \
                 uint64_t dim);                             \
    float                                                   \
    SQ4ComputeL2Sqr(const float* RESTRICT query,            \
                    const uint8_t* RESTRICT codes,          \
                    const float* RESTRICT lower_bound,      \
                    const float* RESTRICT diff,             \
                    uint64_t dim);                          \
    float                                                   \
    SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,       \
                      const uint8_t* RESTRICT codes2,       \
                      const float* RESTRICT lower_bound,    \
                      const float* RESTRICT diff,           \
                      uint64_t dim);                        \
    float                                                   \
    SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,    \
                         const uint8_t* RESTRICT codes2,    \
                         const float* RESTRICT lower_bound, \
                         const float* RESTRICT diff,        \
                         uint64_t dim);                     \
    }  // namespace ns

DECLARE_SQ4_FUNCTIONS(generic)
DECLARE_SQ4_FUNCTIONS(sse)
DECLARE_SQ4_FUNCTIONS(avx)
DECLARE_SQ4_FUNCTIONS(avx2)
DECLARE_SQ4_FUNCTIONS(avx512)
DECLARE_SQ4_FUNCTIONS(neon)
DECLARE_SQ4_FUNCTIONS(sve)

#undef DECLARE_SQ4_FUNCTIONS

using SQ4ComputeType = float (*)(const float* RESTRICT query,
                                 const uint8_t* RESTRICT codes,
                                 const float* RESTRICT lower_bound,
                                 const float* RESTRICT diff,
                                 uint64_t dim);
extern SQ4ComputeType SQ4ComputeIP;
extern SQ4ComputeType SQ4ComputeL2Sqr;

using SQ4ComputeCodesType = float (*)(const uint8_t* RESTRICT codes1,
                                      const uint8_t* RESTRICT codes2,
                                      const float* RESTRICT lower_bound,
                                      const float* RESTRICT diff,
                                      uint64_t dim);
extern SQ4ComputeCodesType SQ4ComputeCodesIP;
extern SQ4ComputeCodesType SQ4ComputeCodesL2Sqr;
}  // namespace vsag
