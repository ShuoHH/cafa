#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rle {

inline uint64_t bits_of(double value) { return std::bit_cast<uint64_t>(value); }

// Elias-gamma code length for n >= 1.
inline uint64_t gamma_bits(uint64_t n) {
    if (n == 0) throw std::invalid_argument("gamma_bits: n must be >= 1");
    return 2 * (63 - std::countl_zero(n)) + 1;
}

class BitWriter {
public:
    void put(uint64_t value, unsigned length) {
        if (length == 0 || length > 64)
            throw std::invalid_argument("BitWriter: invalid bit length");
        for (unsigned i = length; i-- > 0;) put_bit((value >> i) & 1u);
    }

    void put_bit(unsigned bit) {
        if (bits_ % 64 == 0) words_.push_back(0);
        if (bit) words_.back() |= uint64_t{1} << (63 - bits_ % 64);
        ++bits_;
    }

    void gamma(uint64_t value) {
        if (value == 0) throw std::invalid_argument("gamma: value must be >= 1");
        const unsigned high = 63 - std::countl_zero(value);
        for (unsigned i = 0; i < high; ++i) put_bit(0);
        put(value, high + 1);
    }

    uint64_t bits() const { return bits_; }

    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> out;
        out.reserve(words_.size() * sizeof(uint64_t));
        for (uint64_t word : words_)
            for (int i = 7; i >= 0; --i)
                out.push_back(static_cast<uint8_t>(word >> (i * 8)));
        return out;
    }

private:
    std::vector<uint64_t> words_;
    uint64_t bits_ = 0;
};

class BitReader {
public:
    BitReader(const uint8_t *data, size_t nbytes, uint64_t nbits)
        : data_(data), nbytes_(nbytes), nbits_(nbits) {
        if (nbits > std::numeric_limits<uint64_t>::max() - 7 ||
            (nbits + 7) / 8 > nbytes)
            throw std::runtime_error("BitReader: bit count exceeds buffer");
    }

    unsigned bit() {
        if (pos_ >= nbits_)
            throw std::runtime_error("BitReader: read past end of stream");
        const unsigned result = (data_[pos_ / 8] >> (7 - pos_ % 8)) & 1u;
        ++pos_;
        return result;
    }

    uint64_t gamma() {
        unsigned zeros = 0;
        while (bit() == 0) {
            if (++zeros > 63)
                throw std::runtime_error("BitReader: malformed gamma code");
        }
        uint64_t value = 1;
        for (unsigned i = 0; i < zeros; ++i) value = (value << 1) | bit();
        return value;
    }

    uint64_t position() const { return pos_; }

private:
    const uint8_t *data_;
    size_t nbytes_;
    uint64_t nbits_;
    uint64_t pos_ = 0;
};

namespace detail {
inline void put_u64(std::vector<uint8_t> &out, uint64_t value) {
    for (int i = 7; i >= 0; --i)
        out.push_back(static_cast<uint8_t>(value >> (i * 8)));
}

inline uint64_t get_u64(const std::vector<uint8_t> &in, size_t offset) {
    if (offset + sizeof(uint64_t) > in.size())
        throw std::runtime_error("rle: truncated buffer");
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i)
        value = (value << 8) | in[offset + i];
    return value;
}

constexpr uint64_t kMagic = 0x524C453100000000ULL;  // "RLE1"
constexpr size_t kHeader = 7 * sizeof(uint64_t);
}  // namespace detail

struct NoBase {};

// Inner must provide the streaming adapter interface documented in
// neats_adapters.hpp.  The RLE header is a serialization detail; encoded_bits()
// reports only codec stream bits, matching the existing harness accounting.
template <class Inner, class Base = NoBase>
class RleCodec : public Base {
public:
    std::string name() const { return std::string("rle+") + Inner::name; }

    void prefix_bits(const double *x, size_t n, std::vector<uint64_t> &out) {
        out.clear();
        if (n == 0) return;
        out.reserve(n);
        scan(x, n, [&](uint64_t cost) { out.push_back(cost); });
    }

    uint64_t payload_bits(const double *x, size_t n) {
        uint64_t last = 0;
        if (n) scan(x, n, [&](uint64_t cost) { last = cost; });
        return last;
    }

    uint64_t encoded_bits(const double *x, size_t n) {
        const Encoded encoded = build(x, n);
        return encoded.gamma.bits() + encoded.inner_stream_bits;
    }

    uint64_t padded_bits(const double *x, size_t n) {
        const uint64_t bits = encoded_bits(x, n);
        return ((bits + 7) / 8) * 8;
    }

    std::vector<uint8_t> encode(const double *x, size_t n) {
        const Encoded encoded = build(x, n);
        std::vector<uint8_t> out;
        const std::vector<uint8_t> gamma = encoded.gamma.bytes();
        detail::put_u64(out, detail::kMagic);
        detail::put_u64(out, n);
        detail::put_u64(out, encoded.runs);
        detail::put_u64(out, encoded.gamma.bits());
        detail::put_u64(out, gamma.size());
        detail::put_u64(out, encoded.inner_stream_bits);
        detail::put_u64(out, encoded.inner.size());
        out.insert(out.end(), gamma.begin(), gamma.end());
        out.insert(out.end(), encoded.inner.begin(), encoded.inner.end());
        return out;
    }

    std::vector<double> decode(const std::vector<uint8_t> &buffer, size_t n) {
        using detail::get_u64;
        if (get_u64(buffer, 0) != detail::kMagic)
            throw std::runtime_error("rle: bad magic");
        if (get_u64(buffer, 8) != n)
            throw std::runtime_error("rle: value count mismatch");
        const uint64_t runs = get_u64(buffer, 16);
        const uint64_t gamma_bits_count = get_u64(buffer, 24);
        const uint64_t gamma_bytes = get_u64(buffer, 32);
        const uint64_t inner_bits = get_u64(buffer, 40);
        const uint64_t inner_bytes = get_u64(buffer, 48);
        if (buffer.size() < detail::kHeader ||
            gamma_bytes > buffer.size() - detail::kHeader ||
            inner_bytes != buffer.size() - detail::kHeader - gamma_bytes)
            throw std::runtime_error("rle: bad length");
        if (inner_bits > std::numeric_limits<uint64_t>::max() - 7 ||
            (inner_bits + 7) / 8 > inner_bytes)
            throw std::runtime_error("rle: inner bit count exceeds buffer");
        if (n == 0 || runs == 0 || runs > n)
            throw std::runtime_error("rle: bad run count");

        BitReader reader(buffer.data() + detail::kHeader, gamma_bytes,
                         gamma_bits_count);
        std::vector<uint64_t> lengths(runs);
        uint64_t total = 0;
        for (uint64_t &length : lengths) {
            length = reader.gamma();
            if (length == 0 || length > n || total > n - length)
                throw std::runtime_error("rle: run lengths exceed n");
            total += length;
        }
        if (total != n || reader.position() != gamma_bits_count)
            throw std::runtime_error("rle: run lengths inconsistent");

        const std::vector<uint8_t> inner(
            buffer.begin() + detail::kHeader + gamma_bytes, buffer.end());
        const std::vector<double> heads = Inner::decode(inner, inner_bits, runs);
        if (heads.size() != runs)
            throw std::runtime_error("rle: inner decoder returned wrong count");

        std::vector<double> out;
        out.reserve(n);
        for (size_t k = 0; k < lengths.size(); ++k)
            out.insert(out.end(), lengths[k], heads[k]);
        return out;
    }

private:
    struct Encoded {
        BitWriter gamma;
        std::vector<uint8_t> inner;
        uint64_t inner_stream_bits = 0;
        uint64_t runs = 0;
    };

    template <class Emit>
    void scan(const double *x, size_t n, Emit &&emit) {
        Inner inner;
        inner.start(x[0]);
        uint64_t previous = bits_of(x[0]);
        uint64_t run = 1;
        uint64_t closed_gamma = 0;
        emit(inner.payload_bits() + closed_gamma + gamma_bits(run));
        for (size_t i = 1; i < n; ++i) {
            const uint64_t current = bits_of(x[i]);
            if (current == previous) {
                ++run;
            } else {
                closed_gamma += gamma_bits(run);
                inner.add(x[i]);
                run = 1;
                previous = current;
            }
            emit(inner.payload_bits() + closed_gamma + gamma_bits(run));
        }
    }

    Encoded build(const double *x, size_t n) {
        if (n == 0)
            throw std::invalid_argument("rle: empty segments are not supported");
        Encoded encoded;
        Inner inner;
        inner.start(x[0]);
        uint64_t previous = bits_of(x[0]);
        uint64_t run = 1;
        for (size_t i = 1; i < n; ++i) {
            const uint64_t current = bits_of(x[i]);
            if (current == previous) {
                ++run;
            } else {
                encoded.gamma.gamma(run);
                ++encoded.runs;
                inner.add(x[i]);
                run = 1;
                previous = current;
            }
        }
        encoded.gamma.gamma(run);
        ++encoded.runs;
        encoded.inner = inner.finish(encoded.inner_stream_bits);
        return encoded;
    }
};

}  // namespace rle
