// The alphabet, as a bitmask. Three of them, chosen once at startup by --alphabet
// and fixed for the whole run:
//
//   20  the standard amino acids. Gaps, stop codons and X are all missing data:
//       they name no state, so they place no constraint. The default, and the
//       behaviour every output written before 2026-10-30 was produced under.
//   22  the same, plus '-' and '*' as states of their own. Only X is missing.
//   nuc A, C, G and T. Everything else -- N, '-', lowercase, and the IUPAC
//       partial codes R, Y, K, M, S, W, B, D, H, V -- is missing data. A partial
//       code is read as no constraint at all rather than as a two- or three-state
//       mask: sound, since it only ever widens what a node may hold, and it keeps
//       every observation a single state or none, which hartigan.cpp's pinned
//       internal node and placement.cpp's writeLeafSets both assume.
//
// The width is a runtime value, so every loop over states is bounded by
// numStates() and every fixed-size array by kMaxStates. The one thing that must
// not be done is to compile at a fixed 22 and switch only the lookup table:
// allStates() would still carry bits 20 and 21, and hartigan.cpp's
// `count[a] == best - 1` branch would admit '-' and '*' into V2 at every node
// whose best is 1, which silently changes 20-token results.
#pragma once

#include <array>
#include <cstdint>
#include <string>

using AaSet = uint32_t;

inline constexpr int  kMaxStates   = 22;
inline constexpr char kAlphabet20[] = "ACDEFGHIKLMNPQRSTVWY";
inline constexpr char kAlphabet22[] = "ACDEFGHIKLMNPQRSTVWY-*";
inline constexpr char kAlphabetNuc[] = "ACGT";

enum class AlphabetKind { Aa20, Aa22, Nuc };

// Set once by setAlphabet(), before anything parses a tree or a FASTA.
inline int         gNumStates = 20;
inline AaSet       gAllStates = (AaSet(1) << 20) - 1;
inline const char* gAlphabet  = kAlphabet20;
inline std::array<signed char, 256> gAaTable = [] {
    std::array<signed char, 256> t{};
    t.fill(-1);
    for (int i = 0; i < 20; ++i) t[static_cast<unsigned char>(kAlphabet20[i])] = static_cast<signed char>(i);
    return t;
}();

inline void setAlphabet(AlphabetKind kind) {
    gAlphabet  = kind == AlphabetKind::Aa22 ? kAlphabet22
               : kind == AlphabetKind::Nuc  ? kAlphabetNuc
                                            : kAlphabet20;
    gNumStates = kind == AlphabetKind::Aa22 ? 22 : kind == AlphabetKind::Nuc ? 4 : 20;
    gAllStates = (AaSet(1) << gNumStates) - 1;
    gAaTable.fill(-1);
    for (int i = 0; i < gNumStates; ++i) {
        gAaTable[static_cast<unsigned char>(gAlphabet[i])] = static_cast<signed char>(i);
    }
}

inline int                numStates() { return gNumStates; }
inline AaSet              allStates() { return gAllStates; }
inline const signed char* aaTable()   { return gAaTable.data(); }

inline int  aaIndex(char c)   { return gAaTable[static_cast<unsigned char>(c)]; }
inline bool isMissing(char c) { return aaIndex(c) < 0; }

// The constraint an observed character imposes: one state, or none at all.
inline AaSet aaBit(char c) {
    int i = aaIndex(c);
    return i < 0 ? gAllStates : (AaSet(1) << i);
}

inline int setSize(AaSet s) { return __builtin_popcount(s); }

inline std::string setToString(AaSet s) {
    std::string out;
    for (int i = 0; i < gNumStates; ++i)
        if (s & (AaSet(1) << i)) out.push_back(gAlphabet[i]);
    return out;
}
