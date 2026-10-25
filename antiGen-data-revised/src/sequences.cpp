#include "sequences.hpp"

#include "alphabet.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

// --- buffered line reading -------------------------------------------------
// The FASTA is 12 GB, so lines are handed out as pointers into a reusable
// buffer rather than copied into a std::string one at a time.
class LineReader {
  public:
    explicit LineReader(const std::string& path) : buf_(1 << 22) {
        f_ = std::fopen(path.c_str(), "rb");
        if (!f_) throw std::runtime_error("cannot read " + path);
    }
    ~LineReader() { if (f_) std::fclose(f_); }
    LineReader(const LineReader&) = delete;
    LineReader& operator=(const LineReader&) = delete;

    bool next(const char*& p, size_t& n) {
        for (;;) {
            if (begin_ < end_) {
                const char* nl =
                    static_cast<const char*>(std::memchr(buf_.data() + begin_, '\n', end_ - begin_));
                if (nl) {
                    p = buf_.data() + begin_;
                    n = static_cast<size_t>(nl - p);
                    if (n && p[n - 1] == '\r') --n;
                    begin_ = static_cast<size_t>(nl - buf_.data()) + 1;
                    return true;
                }
            }
            if (eof_) {
                if (begin_ >= end_) return false;
                p = buf_.data() + begin_;
                n = end_ - begin_;
                if (n && p[n - 1] == '\r') --n;
                begin_ = end_;
                return true;
            }
            refill();
        }
    }

  private:
    void refill() {
        const size_t live = end_ - begin_;
        if (live) std::memmove(buf_.data(), buf_.data() + begin_, live);
        begin_ = 0;
        end_   = live;
        if (end_ == buf_.size()) buf_.resize(buf_.size() * 2);  // a line longer than the buffer
        const size_t got = std::fread(buf_.data() + end_, 1, buf_.size() - end_, f_);
        end_ += got;
        if (got == 0) eof_ = true;
    }

    std::FILE*        f_ = nullptr;
    std::vector<char> buf_;
    size_t            begin_ = 0, end_ = 0;
    bool              eof_ = false;
};

// --- strain -> date table --------------------------------------------------
// Up to 9.4M entries looked up 9.4M times. std::unordered_map would need a
// std::string per probe under C++17, so this is open addressing over a names
// blob instead. It grows: `reserve` is a starting size, not a bound. Without
// that it does not merely slow down when it fills -- linear probing over a full
// table never terminates, which is what a no-cutoff run found.
class StrainTable {
  public:
    void reserve(size_t n) {
        size_t cap = 16;
        while (cap < n * 2) cap <<= 1;
        slot_.assign(cap, Slot{});
        mask_ = cap - 1;
        names_.reserve(n * 24);
    }

    void insert(const char* s, size_t n, int32_t date) {
        if ((count_ + 1) * 2 > slot_.size()) grow();
        const uint64_t h = hash(s, n);
        size_t i = h & mask_;
        while (slot_[i].len) {
            if (slot_[i].hash == h && slot_[i].len == n &&
                std::memcmp(names_.data() + slot_[i].off, s, n) == 0) {
                return;  // duplicate strain name; keep the first
            }
            i = (i + 1) & mask_;
        }
        slot_[i].hash = h;
        slot_[i].off  = names_.size();
        slot_[i].len  = static_cast<uint32_t>(n);
        slot_[i].date = date;
        names_.append(s, n);
        ++count_;
    }

    // Returns the date, or 0 when the strain is not in the window.
    int32_t find(const char* s, size_t n) const {
        if (!mask_) return 0;
        const uint64_t h = hash(s, n);
        size_t i = h & mask_;
        while (slot_[i].len) {
            if (slot_[i].hash == h && slot_[i].len == n &&
                std::memcmp(names_.data() + slot_[i].off, s, n) == 0) {
                return slot_[i].date;
            }
            i = (i + 1) & mask_;
        }
        return 0;
    }

    size_t size() const { return count_; }

  private:
    // Double and reinsert. A slot carries its own hash and an offset into the
    // names blob, so nothing is re-hashed and nothing is copied but the slots.
    void grow() {
        std::vector<Slot> old;
        old.swap(slot_);
        const size_t cap = old.empty() ? 16 : old.size() * 2;
        slot_.assign(cap, Slot{});
        mask_ = cap - 1;
        for (const Slot& sl : old) {
            if (!sl.len) continue;
            size_t i = sl.hash & mask_;
            while (slot_[i].len) i = (i + 1) & mask_;
            slot_[i] = sl;
        }
    }

    struct Slot {
        uint64_t hash = 0;
        size_t   off  = 0;
        uint32_t len  = 0;
        int32_t  date = 0;
    };

    static uint64_t hash(const char* s, size_t n) {
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<unsigned char>(s[i]);
            h *= 1099511628211ULL;
        }
        return h | 1;  // 0 would collide with an empty slot's sentinel
    }

    std::vector<Slot> slot_;
    std::string       names_;
    size_t            mask_  = 0;
    size_t            count_ = 0;
};

bool isLeapYear(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

size_t firstField(const char* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] == ' ' || p[i] == '\t') return i;
    }
    return n;
}

}  // namespace

int32_t parseDateCode(const char* s, size_t len) {
    if (len != 10 || s[4] != '-' || s[7] != '-') return 0;
    int v[3] = {0, 0, 0};
    const int at[3] = {0, 5, 8};
    const int w[3]  = {4, 2, 2};
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < w[k]; ++i) {
            const char c = s[at[k] + i];
            if (c < '0' || c > '9') return 0;
            v[k] = v[k] * 10 + (c - '0');
        }
    }
    if (v[1] < 1 || v[1] > 12 || v[2] < 1 || v[2] > 31) return 0;
    return v[0] * 10000 + v[1] * 100 + v[2];
}

// Nextstrain's num_date: the midpoint of the collection day, as a fraction of
// the year. Checked against the input tree's floor of 2019.9849 (2019-12-26).
double decimalYear(int32_t dateCode) {
    const int y = dateCode / 10000;
    const int m = (dateCode / 100) % 100;
    const int d = dateCode % 100;
    static const int cum[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
    int day = cum[m - 1] + d;
    if (m > 2 && isLeapYear(y)) ++day;
    return y + (day - 0.5) / (isLeapYear(y) ? 366.0 : 365.0);
}

double decimalYearStart(int32_t dateCode) {
    const int y = dateCode / 10000;
    return decimalYear(dateCode) - 0.5 / (isLeapYear(y) ? 366.0 : 365.0);
}

namespace {

// The smallest divergence whose cumulative count reaches p of the total.
int quantileOf(const std::vector<long long>& counts, double p) {
    long long n = 0;
    for (long long c : counts) n += c;
    if (n == 0) return 0;
    const double k   = p * static_cast<double>(n);
    long long    run = 0;
    for (size_t d = 0; d < counts.size(); ++d) {
        run += counts[d];
        if (static_cast<double>(run) >= k) return static_cast<int>(d);
    }
    return static_cast<int>(counts.size()) - 1;
}

// Per-month divergence histograms -> the envelope. A month too thin to put a
// quantile on borrows from its neighbours, widening until it has clockMinN
// records or runs out of patience; the two months at either end of the data
// and the first weeks of the pandemic are the ones that need it.
ClockEnvelope buildEnvelope(const std::vector<std::vector<long long>>& hist, int firstMonth,
                            const PrepareConfig& cfg) {
    ClockEnvelope env;
    const int n = static_cast<int>(hist.size());
    if (n == 0) return env;
    env.firstMonth = firstMonth;
    env.tolMonths  = cfg.clockTolMonths;
    env.slack      = cfg.clockSlack;
    env.hi.assign(static_cast<size_t>(n), 0);
    env.loSuffixMin.assign(static_cast<size_t>(n), 0);

    std::vector<long long> win;
    for (int i = 0; i < n; ++i) {
        win = hist[static_cast<size_t>(i)];
        long long have = 0;
        for (long long c : win) have += c;
        for (int w = 1; have < cfg.clockMinN && w <= 6; ++w) {
            win.assign(win.size(), 0);
            have = 0;
            for (int j = i - w; j <= i + w; ++j) {
                if (j < 0 || j >= n) continue;
                for (size_t d = 0; d < win.size(); ++d) {
                    win[d] += hist[static_cast<size_t>(j)][d];
                    have += hist[static_cast<size_t>(j)][d];
                }
            }
        }
        env.hi[static_cast<size_t>(i)]          = quantileOf(win, cfg.clockHiQ);
        env.loSuffixMin[static_cast<size_t>(i)] = quantileOf(win, cfg.clockLoQ);
    }
    // The upper curve is made non-decreasing from the right, which is what
    // clips a month whose own tail is mislabelled; the lower one is only
    // turned into suffix minima, since a lingering lineage lowers it for real.
    for (int i = n - 2; i >= 0; --i) {
        env.hi[static_cast<size_t>(i)] =
            std::min(env.hi[static_cast<size_t>(i)], env.hi[static_cast<size_t>(i + 1)]);
        env.loSuffixMin[static_cast<size_t>(i)] =
            std::min(env.loSuffixMin[static_cast<size_t>(i)],
                     env.loSuffixMin[static_cast<size_t>(i + 1)]);
    }
    return env;
}

}  // namespace

int monthIndex(int32_t dateCode) {
    return (dateCode / 10000) * 12 + ((dateCode / 100) % 100) - 1;
}

void ClockEnvelope::bandAt(int month, int& lo, int& hiOut) const {
    const int n = static_cast<int>(hi.size());
    const int u = std::min(std::max(month + tolMonths - firstMonth, 0), n - 1);
    const int l = std::min(std::max(month - tolMonths - firstMonth, 0), n - 1);
    hiOut = hi[static_cast<size_t>(u)] + slack;
    lo    = loSuffixMin[static_cast<size_t>(l)];
}

void writeClockEnvelope(const ClockEnvelope& env, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::fprintf(f, "#tol_months\t%d\n#slack\t%d\n", env.tolMonths, env.slack);
    std::fprintf(f, "year\tmonth\tcurve_hi\tband_lo\tband_hi\n");
    for (size_t i = 0; i < env.hi.size(); ++i) {
        const int m = env.firstMonth + static_cast<int>(i);
        int lo = 0, hi = 0;
        env.bandAt(m, lo, hi);
        std::fprintf(f, "%d\t%d\t%d\t%d\t%d\n", m / 12, m % 12 + 1, env.hi[i], lo, hi);
    }
    if (std::ferror(f)) { std::fclose(f); throw std::runtime_error("write failed: " + path); }
    std::fclose(f);
}

PreparedSet prepareSequences(const PrepareConfig& cfg, PrepareStats* stats,
                             ClockEnvelope* envelope) {
    PrepareStats st;
    PreparedSet  out;
    out.nSites = cfg.nSites;
    if (static_cast<int>(cfg.rootSeq.size()) < cfg.nSites) {
        throw std::runtime_error("root sequence is shorter than the site count");
    }

    // --- the metadata ---------------------------------------------------
    // Every dated strain goes in, not just the ones inside the date window:
    // the envelope is calibrated over the whole dataset so that a cutoff run
    // and a full run apply the identical criterion, which is the point of
    // having one. The window is applied below, once a record's divergence has
    // been counted towards the calibration.
    StrainTable table;
    table.reserve(1 << 20);
    int firstMonth = 0, lastMonth = 0;
    {
        LineReader  r(cfg.metadata);
        const char* p;
        size_t      n;
        bool        first = true;
        while (r.next(p, n)) {
            if (first) {  // header
                first = false;
                if (n >= 6 && std::memcmp(p, "strain", 6) == 0) continue;
            }
            ++st.metadataRows;
            const size_t tab = firstField(p, n);
            if (tab >= n) continue;  // no date column at all
            const int32_t date = parseDateCode(p + tab + 1, n - tab - 1);
            if (!date) continue;
            const int mi = monthIndex(date);
            if (!st.dated || mi < firstMonth) firstMonth = mi;
            if (!st.dated || mi > lastMonth)  lastMonth  = mi;
            ++st.dated;
            if (!cfg.cutoff || date < cfg.cutoff) ++st.beforeCutoff;
            table.insert(p, tab, date);
        }
    }
    const int nMonths = st.dated ? lastMonth - firstMonth + 1 : 0;

    // --- one pass over the FASTA ---------------------------------------
    out.blob.reserve(static_cast<size_t>(st.beforeCutoff) * cfg.nSites);
    out.rec.reserve(static_cast<size_t>(st.beforeCutoff));

    const char* root = cfg.rootSeq.data();
    std::string strain;
    std::string seq;
    int32_t     date = 0;

    // Divergence histograms, one per month over every record that clears
    // coverage -- the calibration set. A record that fails coverage is missing
    // too much for its divergence to mean anything, so it counts for nothing
    // here either.
    std::vector<std::vector<long long>> hist;
    if (cfg.useClock) {
        hist.assign(static_cast<size_t>(nMonths),
                    std::vector<long long>(static_cast<size_t>(cfg.nSites) + 1, 0));
    }
    // The divergence of each buffered record, parallel to out.rec: the clock
    // test needs the envelope, and the envelope is not known until the pass is
    // over.
    std::vector<int> dRootOf;

    auto flush = [&]() {
        if (strain.empty()) return;
        if (date) {
            ++st.matched;
            const int len = static_cast<int>(seq.size());
            if (len != cfg.nSites && len != cfg.nSites + 1) {
                ++st.badLength;
            } else {
                int covered = 0, dRoot = 0;
                for (int s = 0; s < cfg.nSites; ++s) {
                    if (isMissing(seq[s])) continue;
                    ++covered;
                    if (seq[s] != root[s]) ++dRoot;
                }
                const bool inWindow = !cfg.cutoff || date < cfg.cutoff;
                if (covered < cfg.minCovered) {
                    if (inWindow) ++st.failCoverage;
                } else {
                    if (cfg.useClock) {
                        hist[static_cast<size_t>(monthIndex(date) - firstMonth)]
                            [static_cast<size_t>(dRoot)] += 1;
                    }
                    if (inWindow) {
                        PreparedSet::Record rc;
                        rc.strain   = strain;
                        rc.dateCode = date;
                        rc.offset   = out.blob.size();
                        out.blob.append(seq, 0, static_cast<size_t>(cfg.nSites));
                        out.rec.push_back(std::move(rc));
                        dRootOf.push_back(dRoot);
                    }
                }
            }
        }
        strain.clear();
        seq.clear();
        date = 0;
    };

    {
        LineReader  r(cfg.fasta);
        const char* p;
        size_t      n;
        while (r.next(p, n)) {
            if (n && p[0] == '>') {
                flush();
                ++st.fastaRecords;
                const size_t end = firstField(p + 1, n - 1);
                strain.assign(p + 1, end);
                date = table.find(strain.data(), strain.size());
            } else if (date) {
                // Unmatched records are the overwhelming majority; not copying
                // their residues is most of the pass's speed.
                seq.append(p, n);
            }
        }
        flush();
    }

    // --- the clock, now that the envelope can be built -------------------
    // The rejected sequences stay in the blob as dead space rather than being
    // compacted out of it: there are a few hundred of them against millions of
    // records, and nothing reads the blob except through out.rec.
    ClockEnvelope env;
    if (cfg.useClock && nMonths > 0) {
        env = buildEnvelope(hist, firstMonth, cfg);
        size_t w = 0;
        for (size_t i = 0; i < out.rec.size(); ++i) {
            int lo = 0, hi = 0;
            env.bandAt(monthIndex(out.rec[i].dateCode), lo, hi);
            if (dRootOf[i] > hi)      { ++st.failClockHigh; continue; }
            if (dRootOf[i] < lo)      { ++st.failClockLow;  continue; }
            if (w != i) out.rec[w] = std::move(out.rec[i]);
            ++w;
        }
        out.rec.resize(w);
    }
    if (envelope) *envelope = env;

    for (const PreparedSet::Record& rc : out.rec) {
        ++st.kept;
        int covered = 0;
        for (int s = 0; s < cfg.nSites; ++s) {
            if (!isMissing(out.blob[rc.offset + static_cast<size_t>(s)])) ++covered;
        }
        st.missingTotal += cfg.nSites - covered;
        if (covered == cfg.nSites) ++st.fullyCovered;
    }

    // Date first, then strain name: a total order, so the file is reproducible.
    // The within-day shuffle is `place`'s job, not this pass's.
    std::sort(out.rec.begin(), out.rec.end(),
              [](const PreparedSet::Record& a, const PreparedSet::Record& b) {
                  if (a.dateCode != b.dateCode) return a.dateCode < b.dateCode;
                  return a.strain < b.strain;
              });

    if (stats) *stats = st;
    return out;
}

void writePrepared(const PreparedSet& set, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::vector<char> buf(1 << 22);
    std::setvbuf(f, buf.data(), _IOFBF, buf.size());

    std::fprintf(f, "#sites\t%d\n", set.nSites);
    std::fprintf(f, "#count\t%zu\n", set.size());
    std::fprintf(f, "#strain\tdate\tseq\n");
    for (size_t i = 0; i < set.size(); ++i) {
        const PreparedSet::Record& rc = set.rec[i];
        std::fprintf(f, "%s\t%04d-%02d-%02d\t", rc.strain.c_str(), rc.dateCode / 10000,
                     (rc.dateCode / 100) % 100, rc.dateCode % 100);
        std::fwrite(set.seq(i), 1, static_cast<size_t>(set.nSites), f);
        std::fputc('\n', f);
    }
    if (std::ferror(f)) { std::fclose(f); throw std::runtime_error("write failed: " + path); }
    std::fclose(f);
}

PreparedSet readPrepared(const std::string& path) {
    PreparedSet out;
    LineReader  r(path);
    const char* p;
    size_t      n;
    size_t      expect = 0;

    while (r.next(p, n)) {
        if (n && p[0] == '#') {
            if (n > 7 && std::memcmp(p, "#sites\t", 7) == 0) {
                out.nSites = std::atoi(std::string(p + 7, n - 7).c_str());
            } else if (n > 7 && std::memcmp(p, "#count\t", 7) == 0) {
                expect = static_cast<size_t>(std::atoll(std::string(p + 7, n - 7).c_str()));
                out.blob.reserve(expect * static_cast<size_t>(out.nSites));
                out.rec.reserve(expect);
            }
            continue;
        }
        if (!n) continue;
        if (out.nSites <= 0) throw std::runtime_error("no #sites header in " + path);

        const size_t t1 = firstField(p, n);
        if (t1 >= n) throw std::runtime_error("malformed record in " + path);
        const size_t t2 = t1 + 1 + firstField(p + t1 + 1, n - t1 - 1);
        if (t2 >= n) throw std::runtime_error("malformed record in " + path);
        const size_t seqLen = n - t2 - 1;
        if (static_cast<int>(seqLen) != out.nSites) {
            throw std::runtime_error("record length does not match #sites in " + path);
        }

        PreparedSet::Record rc;
        rc.strain.assign(p, t1);
        rc.dateCode = parseDateCode(p + t1 + 1, t2 - t1 - 1);
        if (!rc.dateCode) throw std::runtime_error("malformed date in " + path);
        rc.offset = out.blob.size();
        out.blob.append(p + t2 + 1, seqLen);
        out.rec.push_back(std::move(rc));
    }
    if (expect && out.rec.size() != expect) {
        throw std::runtime_error("record count does not match the #count header in " + path);
    }
    return out;
}

// --- the position count matrix ---------------------------------------------

void ResidueCounts::init(int sites) {
    nSites = sites;
    n.assign(static_cast<size_t>(sites) * kNumStates, 0);
    sequences = 0;
    residues  = 0;
}

void ResidueCounts::add(const char* seq) {
    const signed char* tbl = aaTable();
    for (int s = 0; s < nSites; ++s) {
        const int a = tbl[static_cast<unsigned char>(seq[s])];
        if (a < 0) continue;  // '-', '*', 'X': no state, so no count
        ++n[static_cast<size_t>(s) * kNumStates + static_cast<size_t>(a)];
        ++residues;
    }
    ++sequences;
}

void writeResidueCounts(const ResidueCounts& counts, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);

    std::fprintf(f, "#sequences\t%lld\n", counts.sequences);
    std::fprintf(f, "#sites\t%d\n", counts.nSites);
    std::fprintf(f, "#site");
    for (int a = 0; a < kNumStates; ++a) std::fprintf(f, "\t%c", kAlphabet[a]);
    std::fputc('\n', f);
    for (int s = 0; s < counts.nSites; ++s) {
        std::fprintf(f, "%d", s + 1);
        for (int a = 0; a < kNumStates; ++a) {
            std::fprintf(f, "\t%lld", counts.n[static_cast<size_t>(s) * kNumStates + static_cast<size_t>(a)]);
        }
        std::fputc('\n', f);
    }
    if (std::ferror(f)) { std::fclose(f); throw std::runtime_error("write failed: " + path); }
    std::fclose(f);
}

// --- the alignment of the kept sequences ------------------------------------

void writeMsaRecord(std::FILE* f, const std::string& strain, const char* seq, int nSites) {
    std::fputc('>', f);
    std::fwrite(strain.data(), 1, strain.size(), f);
    std::fputc('\n', f);
    std::fwrite(seq, 1, static_cast<size_t>(nSites), f);
    std::fputc('\n', f);
}

void writeMsa(const PreparedSet& set, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::vector<char> buf(1 << 22);
    std::setvbuf(f, buf.data(), _IOFBF, buf.size());

    for (size_t i = 0; i < set.size(); ++i) writeMsaRecord(f, set.rec[i].strain, set.seq(i), set.nSites);

    if (std::ferror(f)) { std::fclose(f); throw std::runtime_error("write failed: " + path); }
    std::fclose(f);
}
