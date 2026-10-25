// The 20 standard amino acids as a bitmask. Gaps, stop codons and X are all
// missing data: they name no state, so they place no constraint.
#pragma once

#include <array>
#include <cstdint>
#include <string>

using AaSet = uint32_t;

inline constexpr int   kNumStates = 20;
inline constexpr char  kAlphabet[] = "ACDEFGHIKLMNPQRSTVWY";
inline constexpr AaSet kAllStates = (AaSet(1) << kNumStates) - 1;

inline const signed char* aaTable() {
    static const std::array<signed char, 256> table = [] {
        std::array<signed char, 256> t{};
        t.fill(-1);
        for (int i = 0; i < kNumStates; ++i) t[static_cast<unsigned char>(kAlphabet[i])] = static_cast<signed char>(i);
        return t;
    }();
    return table.data();
}

inline int  aaIndex(char c)   { return aaTable()[static_cast<unsigned char>(c)]; }
inline bool isMissing(char c) { return aaIndex(c) < 0; }

// The constraint an observed character imposes: one state, or none at all.
inline AaSet aaBit(char c) {
    int i = aaIndex(c);
    return i < 0 ? kAllStates : (AaSet(1) << i);
}

inline int setSize(AaSet s) { return __builtin_popcount(s); }

inline std::string setToString(AaSet s) {
    std::string out;
    for (int i = 0; i < kNumStates; ++i)
        if (s & (AaSet(1) << i)) out.push_back(kAlphabet[i]);
    return out;
}
