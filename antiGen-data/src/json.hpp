// Minimal pull-style JSON reader. Enough to walk a Nextstrain v2 dataset and
// pick out the few fields we need without materialising the whole document.
#pragma once

#include <string>

struct JsonReader {
    const char* p;
    const char* end;

    JsonReader(const char* begin, const char* finish) : p(begin), end(finish) {}
    explicit JsonReader(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    void skipWs();
    char peek();                 // throws at end of input
    void expect(char c);         // skips whitespace, then consumes c
    bool tryConsume(char c);     // skips whitespace, consumes c only if it matches
    std::string readString();    // skips whitespace, reads a quoted string
    double      readNumber();    // skips whitespace, reads a number (null -> NaN)
    void skipValue();            // skips any value, however nested

    [[noreturn]] void fail(const std::string& what) const;
};

// Walks an object, calling f(key) with the value positioned ready to read.
// f must consume exactly one value (call r.skipValue() to discard it).
template <class F>
void forEachMember(JsonReader& r, F&& f) {
    r.expect('{');
    if (r.tryConsume('}')) return;
    for (;;) {
        std::string key = r.readString();
        r.expect(':');
        f(key);
        if (r.tryConsume(',')) continue;
        r.expect('}');
        return;
    }
}

// Walks an array, calling f() with each element positioned ready to read.
template <class F>
void forEachElement(JsonReader& r, F&& f) {
    r.expect('[');
    if (r.tryConsume(']')) return;
    for (;;) {
        f();
        if (r.tryConsume(',')) continue;
        r.expect(']');
        return;
    }
}

std::string readWholeFile(const std::string& path);
