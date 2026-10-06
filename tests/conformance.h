// The spec's conformance fixtures, read from the JSON copies in tests/conformance, which CI keeps
// byte identical to tandem-spec. A JSON reader for those files, the case records, and the SHA-256
// and FNV-1a hashes of hashes.json.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace conf {

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    std::string s; // a string, or a number's text
    std::vector<Json> a;        // array elements, or object values
    std::vector<std::string> k; // object keys, a vector of pairs would need Json complete

    const Json *find(const char *key) const {
        for (size_t i = 0; i < k.size(); i++)
            if (k[i] == key) return &a[i];
        return nullptr;
    }
    const Json &operator[](const char *key) const {
        const Json *v = find(key);
        if (!v) {
            std::printf("FAIL conformance: no field %s\n", key);
            std::exit(1);
        }
        return *v;
    }
    uint64_t u() const { return std::strtoull(s.c_str(), nullptr, 10); }
    uint64_t hex() const { return std::strtoull(s.c_str(), nullptr, 16); }
};

// The fixtures hold no escapes beyond plain ASCII strings, so the reader skips none.
struct Reader {
    const char *p;
    void ws() {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    }
    std::string str() {
        const char *b = ++p;
        while (*p != '"') p += *p == '\\' ? 2 : 1;
        return std::string(b, p++);
    }
    Json value() {
        Json v;
        ws();
        if (*p == '{') {
            v.type = Json::Obj;
            p++;
            for (ws(); *p != '}'; ws()) {
                v.k.push_back(str());
                ws();
                p++; // ':'
                v.a.push_back(value());
                ws();
                if (*p == ',') p++, ws();
            }
            p++;
        } else if (*p == '[') {
            v.type = Json::Arr;
            p++;
            for (ws(); *p != ']'; ws()) {
                v.a.push_back(value());
                ws();
                if (*p == ',') p++, ws();
            }
            p++;
        } else if (*p == '"') {
            v.type = Json::Str;
            v.s = str();
        } else if (*p == 't' || *p == 'f' || *p == 'n') {
            v.type = *p == 'n' ? Json::Null : Json::Bool;
            v.s = *p == 't' ? "true" : "false";
            p += *p == 'f' ? 5 : 4;
        } else {
            v.type = Json::Num;
            const char *b = p;
            while (*p && std::strchr("+-.0123456789eE", *p)) p++;
            v.s.assign(b, p);
        }
        return v;
    }
};

inline Json load(const char *dir, const char *name) {
    char path[512];
    std::snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::printf("FAIL cannot open %s\n", path);
        std::exit(1);
    }
    std::string text;
    char buf[65536];
    for (size_t k; (k = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, k);
    std::fclose(f);
    Reader r{text.c_str()};
    return r.value();
}

// One case of below, fill_below, normal, exponential or choice.json. `values` holds bit patterns.
struct Case {
    std::string id, kind;
    uint32_t key[4];
    uint32_t K;
    uint64_t start, end = 0, range = 0, capacity = 0;
    bool has_end = false;
    size_t n;
    int rejected = -1;
    std::vector<uint64_t> values, cut;
    std::vector<uint32_t> alias;
    std::vector<double> weights;

    bool is(const char *suffix) const {
        size_t k = std::strlen(suffix);
        return id.size() >= k && id.compare(id.size() - k, k, suffix) == 0;
    }
    double f64(size_t i) const {
        double x;
        std::memcpy(&x, &values[i], 8);
        return x;
    }
    float f32(size_t i) const {
        uint32_t b = (uint32_t)values[i];
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    }
};

inline std::vector<Case> cases(const char *dir, const char *name) {
    Json j = load(dir, name);
    std::vector<Case> out;
    for (const Json &c : j["cases"].a) {
        Case k;
        k.id = c["id"].s;
        k.kind = c["kind"].s;
        for (int w = 0; w < 4; w++) k.key[w] = (uint32_t)c["key"].a[w].hex();
        k.K = (uint32_t)c["K"].u();
        k.start = c["start"].u();
        k.n = (size_t)c["n"].u();
        if (const Json *e = c.find("end")) k.end = e->u(), k.has_end = true;
        if (const Json *r = c.find("range")) k.range = r->hex();
        if (const Json *r = c.find("rejected")) k.rejected = (int)r->u();
        if (const Json *s = c.find("capacity")) k.capacity = s->hex();
        for (const Json &v : c["values"].a) k.values.push_back(v.hex());
        if (const Json *w = c.find("weights"))
            for (const Json &v : w->a) {
                uint64_t b = v.hex();
                double x;
                std::memcpy(&x, &b, 8);
                k.weights.push_back(x);
            }
        if (const Json *t = c.find("cut"))
            for (const Json &v : t->a) k.cut.push_back(v.hex());
        if (const Json *t = c.find("alias"))
            for (const Json &v : t->a) k.alias.push_back((uint32_t)v.hex());
        out.push_back(k);
    }
    return out;
}

inline const Case &find(const std::vector<Case> &cs, const char *suffix) {
    for (const Case &c : cs)
        if (c.is(suffix)) return c;
    std::printf("FAIL conformance: no case %s\n", suffix);
    std::exit(1);
}

inline uint64_t fnv1a(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}
constexpr uint64_t FNV_BASIS = 0xcbf29ce484222325ull;

// SHA-256 (FIPS 180-4) of a byte string, as lowercase hexadecimal.
inline std::string sha256(const void *data, size_t n) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](uint32_t x, int r) { return (x >> r) | (x << (32 - r)); };
    std::vector<unsigned char> m(static_cast<const unsigned char *>(data),
                                 static_cast<const unsigned char *>(data) + n);
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 7; i >= 0; i--) m.push_back((unsigned char)((uint64_t)n * 8 >> (8 * i)));
    for (size_t b = 0; b < m.size(); b += 64) {
        uint32_t w[64];
        for (int t = 0; t < 16; t++)
            w[t] = (uint32_t)m[b + 4 * t] << 24 | (uint32_t)m[b + 4 * t + 1] << 16 |
                   (uint32_t)m[b + 4 * t + 2] << 8 | m[b + 4 * t + 3];
        for (int t = 16; t < 64; t++) {
            uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
            uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        uint32_t v[8];
        std::memcpy(v, h, sizeof v);
        for (int t = 0; t < 64; t++) {
            uint32_t t1 = v[7] + (rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25)) +
                          ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[t] + w[t];
            uint32_t t2 = (rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22)) +
                          ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
            std::memmove(v + 1, v, 7 * sizeof v[0]);
            v[4] += t1;
            v[0] = t1 + t2;
        }
        for (int i = 0; i < 8; i++) h[i] += v[i];
    }
    char hex[65];
    for (int i = 0; i < 8; i++) std::snprintf(hex + 8 * i, 9, "%08x", h[i]);
    return std::string(hex, 64);
}

// The stream of hashes.json whose file name ends in `name`.
inline const Json &stream(const Json &hashes, const char *name) {
    for (const Json &s : hashes["streams"].a)
        if (s["file"].s.size() >= std::strlen(name) &&
            s["file"].s.compare(s["file"].s.size() - std::strlen(name), std::string::npos, name) == 0)
            return s;
    std::printf("FAIL conformance: no stream %s\n", name);
    std::exit(1);
}

} // namespace conf
