#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// codec_harness.cpp includes these external legacy codec headers first.
// Those files predate include guards, so do not include them a second time in
// that translation unit. Standalone users (RleLayerTest) still include them
// here normally.
#ifndef NEATS_ADAPTERS_CODEC_HEADERS_INCLUDED
#include "Gorilla/CompressorGorilla.hpp"
#include "Gorilla/DecompressorGorilla.hpp"
#include "Chimp/CompressorChimp.hpp"
#include "Chimp/DecompressorChimp.hpp"
#include "Chimp128/CompressorChimp128.hpp"
#include "Chimp128/DecompressorChimp128.hpp"
#include "TSXor/CompressorTSXor.hpp"
#include "TSXor/DecompressorTSXor.hpp"
#endif

namespace neats_adapters {

// Preserve the benchmark BitStream buckets verbatim.  In particular, the
// final partial bucket is right-aligned in the legacy representation.
inline std::vector<uint8_t> bitstream_bytes(const BitStream &bs) {
    std::vector<uint8_t> out;
    out.reserve(bs.data.size() * sizeof(uint64_t));
    for (const uint64_t word : bs.data)
        for (int i = 7; i >= 0; --i)
            out.push_back(static_cast<uint8_t>(word >> (i * 8)));
    return out;
}

inline BitStream bitstream_from_bytes(const std::vector<uint8_t> &bytes,
                                      uint64_t bits) {
    if (bytes.size() % sizeof(uint64_t) != 0)
        throw std::runtime_error("bitstream buffer not word-aligned");
    std::deque<uint64_t> words;
    for (size_t off = 0; off < bytes.size(); off += sizeof(uint64_t)) {
        uint64_t word = 0;
        for (size_t i = 0; i < sizeof(uint64_t); ++i)
            word = (word << 8) | bytes[off + i];
        words.push_back(word);
    }
    // The legacy decoders do not bounds-check BitStream::get().  Keep a
    // finite zero sentinel after the serialized buckets.
    for (int i = 0; i < 128; ++i) words.push_back(0);
    if (words.empty()) words.push_back(0);
    return BitStream(words, bits);
}

template <class Decoder>
std::vector<double> drain(Decoder &decoder, size_t count, const char *who) {
    if (count == 0) throw std::invalid_argument(std::string(who) + ": empty stream");
    std::vector<double> out;
    out.reserve(count);
    out.push_back(decoder.storedValue);
    for (size_t i = 1; i < count; ++i) {
        if (!decoder.hasNext())
            throw std::runtime_error(std::string(who) + ": stream ended early");
        out.push_back(decoder.storedValue);
    }
    return out;
}

struct Gorilla {
    static constexpr const char *name = "gorilla";
    std::optional<CompressorGorilla<double>> compressor;
    void start(double value) { compressor.emplace(value); }
    void add(double value) { compressor->addValue(value); }
    uint64_t payload_bits() { return compressor->bs_values.size(); }
    std::vector<uint8_t> finish(uint64_t &stream_bits) {
        compressor->close();
        stream_bits = compressor->bs_values.size();
        return bitstream_bytes(compressor->bs_values);
    }
    static std::vector<double> decode(const std::vector<uint8_t> &bytes,
                                      uint64_t bits, size_t count) {
        BitStream stream = bitstream_from_bytes(bytes, bits);
        DecompressorGorilla<double> decoder(stream, count);
        return drain(decoder, count, name);
    }
};

struct Chimp {
    static constexpr const char *name = "chimp";
    std::optional<CompressorChimp<double>> compressor;
    void start(double value) { compressor.emplace(value); }
    void add(double value) { compressor->addValue(value); }
    uint64_t payload_bits() { return compressor->getSize(); }
    std::vector<uint8_t> finish(uint64_t &stream_bits) {
        compressor->close();
        const BitStream stream = compressor->getBuffer();
        stream_bits = stream.size();
        return bitstream_bytes(stream);
    }
    static std::vector<double> decode(const std::vector<uint8_t> &bytes,
                                      uint64_t bits, size_t count) {
        BitStream stream = bitstream_from_bytes(bytes, bits);
        DecompressorChimp<double> decoder(stream, count);
        return drain(decoder, count, name);
    }
};

struct Chimp128 {
    static constexpr const char *name = "chimp128";
    std::optional<CompressorChimp128<double>> compressor;
    void start(double value) { compressor.emplace(value); }
    void add(double value) { compressor->addValue(value); }
    uint64_t payload_bits() { return compressor->getSize(); }
    std::vector<uint8_t> finish(uint64_t &stream_bits) {
        compressor->close();
        const BitStream stream = compressor->getBuffer();
        stream_bits = stream.size();
        return bitstream_bytes(stream);
    }
    static std::vector<double> decode(const std::vector<uint8_t> &bytes,
                                      uint64_t bits, size_t count) {
        BitStream stream = bitstream_from_bytes(bytes, bits);
        DecompressorChimp128<double> decoder(stream, count);
        return drain(decoder, count, name);
    }
};

struct TSXor {
    static constexpr const char *name = "tsxor";
    std::optional<CompressorTSXor<double>> compressor;
    void start(double value) { compressor.emplace(value); }
    void add(double value) { compressor->addValue(value); }
    uint64_t payload_bits() { return compressor->getSize(); }
    std::vector<uint8_t> finish(uint64_t &stream_bits) {
        compressor->close();
        stream_bits = compressor->getSize();
        return compressor->bytes;
    }
    static std::vector<double> decode(const std::vector<uint8_t> &bytes,
                                      uint64_t, size_t count) {
        DecompressorTSXor<double> decoder(bytes, count);
        return drain(decoder, count, name);
    }
};

}  // namespace neats_adapters
