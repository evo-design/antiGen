#include "json.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

void JsonReader::fail(const std::string& what) const {
    throw std::runtime_error("JSON parse error: " + what);
}

void JsonReader::skipWs() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
}

char JsonReader::peek() {
    skipWs();
    if (p >= end) fail("unexpected end of input");
    return *p;
}

void JsonReader::expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "' but found '" + *p + "'");
    ++p;
}

bool JsonReader::tryConsume(char c) {
    if (peek() != c) return false;
    ++p;
    return true;
}

static void appendUtf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string JsonReader::readString() {
    expect('"');
    std::string out;
    while (p < end) {
        char c = *p++;
        if (c == '"') return out;
        if (c != '\\') { out.push_back(c); continue; }
        if (p >= end) fail("unterminated escape");
        char e = *p++;
        switch (e) {
            case '"':  out.push_back('"');  break;
            case '\\': out.push_back('\\'); break;
            case '/':  out.push_back('/');  break;
            case 'b':  out.push_back('\b'); break;
            case 'f':  out.push_back('\f'); break;
            case 'n':  out.push_back('\n'); break;
            case 'r':  out.push_back('\r'); break;
            case 't':  out.push_back('\t'); break;
            case 'u': {
                if (end - p < 4) fail("truncated \\u escape");
                unsigned cp = 0;
                for (int i = 0; i < 4; ++i) {
                    char h = *p++;
                    cp <<= 4;
                    if (h >= '0' && h <= '9')      cp |= unsigned(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= unsigned(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= unsigned(h - 'A' + 10);
                    else fail("bad hex digit in \\u escape");
                }
                appendUtf8(out, cp);
                break;
            }
            default: fail(std::string("unknown escape '\\") + e + "'");
        }
    }
    fail("unterminated string");
}

// A JSON number, or null, which Nextstrain uses for an absent attribute.
double JsonReader::readNumber() {
    skipWs();
    const char* start = p;
    while (p < end && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
        ++p;
    }
    if (p == start) fail("expected a number");
    const std::string text(start, p);
    if (text == "null") return std::numeric_limits<double>::quiet_NaN();
    return std::strtod(text.c_str(), nullptr);
}

void JsonReader::skipValue() {
    char c = peek();
    if (c == '"') { readString(); return; }
    if (c == '{') {
        forEachMember(*this, [&](const std::string&) { skipValue(); });
        return;
    }
    if (c == '[') {
        forEachElement(*this, [&]() { skipValue(); });
        return;
    }
    // number, true, false or null: run to the next structural character
    const char* start = p;
    while (p < end && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
        ++p;
    }
    if (p == start) fail(std::string("unexpected character '") + c + "'");
}

std::string readWholeFile(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string out;
    std::vector<char> buf(1 << 20);
    for (;;) {
        size_t n = std::fread(buf.data(), 1, buf.size(), f);
        if (n == 0) break;
        out.append(buf.data(), n);
    }
    std::fclose(f);
    return out;
}
