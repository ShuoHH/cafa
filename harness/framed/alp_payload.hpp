#pragma once

// A fully persisted, versioned ALP/ALP-RD adapter format. It is deliberately
// labelled as adapter storage, NOT the upstream algorithmic payload bit count.
#include <alp.hpp>
#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace framed {
using Bytes = std::vector<std::uint8_t>;
inline void put(Bytes& bytes, std::uint64_t value, unsigned width) {
    for (unsigned i = width; i; --i) bytes.push_back(static_cast<std::uint8_t>(value >> ((i - 1) * 8)));
}
class Reader {
    const Bytes& bytes_;
    std::size_t offset_ = 0;
public:
    explicit Reader(const Bytes& bytes) : bytes_(bytes) {}
    std::uint64_t get(unsigned width) {
        if (width > 8 || offset_ > bytes_.size() || width > bytes_.size() - offset_)
            throw std::runtime_error("truncated persisted payload");
        std::uint64_t value = 0;
        while (width--) value = (value << 8) | bytes_[offset_++];
        return value;
    }
    void finish() const { if (offset_ != bytes_.size()) throw std::runtime_error("trailing persisted payload bytes"); }
};
inline constexpr std::size_t vector_size = alp::config::VECTOR_SIZE;
inline void validate_alp_count(std::size_t count) {
    if (!count || count % vector_size || count > alp::config::ROWGROUP_SIZE)
        throw std::runtime_error("ALP frozen segment must be 1024-multiple, at most 102400 values; no implicit tail is permitted");
}
inline void validate_rd(const alp::state<double>& state) {
    if (!state.actual_dictionary_size || state.actual_dictionary_size > alp::config::MAX_RD_DICTIONARY_SIZE ||
        !state.right_bit_width || state.right_bit_width >= 64 || !state.left_bit_width ||
        state.left_bit_width > alp::config::MAX_RD_DICT_BIT_WIDTH || state.left_for_base || state.right_for_base)
        throw std::runtime_error("invalid persisted ALP-RD dictionary/packing parameters");
}
inline Bytes encode_alp(const double* values, std::size_t count) {
    validate_alp_count(count);
    std::vector<double> sample(alp::config::ROWGROUP_SIZE);
    alp::state<double> state{};
    alp::encoder<double>::init(values, 0, count, sample.data(), state);
    if (state.scheme == alp::Scheme::ALP_RD)
        alp::rd_encoder<double>::init(values, 0, count, sample.data(), state);
    else if (state.scheme != alp::Scheme::ALP)
        throw std::runtime_error("ALP selected unsupported scheme");
    Bytes bytes{'A','L','P','F','0','1',static_cast<std::uint8_t>(state.scheme),0};
    if (state.scheme == alp::Scheme::ALP_RD) {
        validate_rd(state);
        put(bytes, state.right_bit_width, 1); put(bytes, state.left_bit_width, 1);
        put(bytes, state.actual_dictionary_size, 1);
        for (auto item : state.left_parts_dict) put(bytes, item, 2);
        for (std::size_t offset = 0; offset < count; offset += vector_size) {
            std::array<std::uint64_t, vector_size> right{}, packed_right{};
            std::array<std::uint16_t, vector_size> left{}, packed_left{}, exceptions{}, positions{}, counts{};
            alp::rd_encoder<double>::encode(values + offset, exceptions.data(), positions.data(), counts.data(),
                                            right.data(), left.data(), state);
            if (counts[0] > vector_size) throw std::runtime_error("ALP-RD exception count overflow");
            std::array<bool, vector_size> seen{};
            for (std::uint16_t i = 0; i < counts[0]; ++i) {
                if (positions[i] >= vector_size || seen[positions[i]]) throw std::runtime_error("invalid ALP-RD exception position");
                seen[positions[i]] = true;
                left[positions[i]] = 0; // upstream placeholder must not index beyond the persisted dictionary
            }
            for (std::size_t i = 0; i < vector_size; ++i)
                if (!seen[i] && left[i] >= state.actual_dictionary_size) throw std::runtime_error("ALP-RD dictionary index overflow");
            fastlanes::generated::ffor::fallback::scalar::ffor(right.data(), packed_right.data(), state.right_bit_width, &state.right_for_base);
            fastlanes::generated::ffor::fallback::scalar::ffor(left.data(), packed_left.data(), state.left_bit_width, &state.left_for_base);
            put(bytes, counts[0], 2);
            for (std::size_t i = 0; i < state.right_bit_width * (vector_size / 64); ++i) put(bytes, packed_right[i], 8);
            for (std::size_t i = 0; i < state.left_bit_width * (vector_size / 16); ++i) put(bytes, packed_left[i], 2);
            for (std::uint16_t i = 0; i < counts[0]; ++i) { put(bytes, positions[i], 2); put(bytes, exceptions[i], 2); }
        }
    } else {
        for (std::size_t offset = 0; offset < count; offset += vector_size) {
            std::array<double, vector_size> exceptions{};
            std::array<std::uint16_t, vector_size> positions{}, counts{};
            std::array<std::int64_t, vector_size> encoded{}, base{}, packed{};
            alp::encoder<double>::encode(values + offset, exceptions.data(), positions.data(), counts.data(), encoded.data(), state);
            alp::bw_t width = 0;
            alp::encoder<double>::analyze_ffor(encoded.data(), width, base.data());
            if (width > 64 || counts[0] > vector_size) throw std::runtime_error("invalid ALP vector packing parameters");
            fastlanes::generated::ffor::fallback::scalar::ffor(encoded.data(), packed.data(), width, base.data());
            put(bytes, width, 1); put(bytes, state.fac, 1); put(bytes, state.exp, 1);
            put(bytes, std::bit_cast<std::uint64_t>(base[0]), 8); put(bytes, counts[0], 2);
            for (std::size_t i = 0; i < width * (vector_size / 64); ++i) put(bytes, std::bit_cast<std::uint64_t>(packed[i]), 8);
            for (std::uint16_t i = 0; i < counts[0]; ++i) {
                put(bytes, positions[i], 2); put(bytes, std::bit_cast<std::uint64_t>(exceptions[i]), 8);
            }
        }
    }
    return bytes;
}

inline std::vector<double> decode_alp(const Bytes& bytes, std::size_t count) {
    validate_alp_count(count);
    Reader reader(bytes);
    for (char ch : std::string("ALPF01")) if (reader.get(1) != static_cast<unsigned char>(ch))
        throw std::runtime_error("invalid persisted ALP magic/version");
    const auto scheme = static_cast<alp::Scheme>(reader.get(1));
    if (reader.get(1)) throw std::runtime_error("nonzero ALP reserved byte");
    std::vector<double> result(count);
    if (scheme == alp::Scheme::ALP_RD) {
        alp::state<double> state{};
        state.scheme = scheme;
        state.right_bit_width = static_cast<alp::bw_t>(reader.get(1));
        state.left_bit_width = static_cast<alp::bw_t>(reader.get(1));
        state.actual_dictionary_size = static_cast<std::uint8_t>(reader.get(1));
        validate_rd(state);
        for (auto& item : state.left_parts_dict) item = static_cast<std::uint16_t>(reader.get(2));
        for (std::size_t offset = 0; offset < count; offset += vector_size) {
            auto exceptions_count = static_cast<std::uint16_t>(reader.get(2));
            if (exceptions_count > vector_size) throw std::runtime_error("invalid persisted ALP-RD exception count");
            std::array<std::uint64_t, vector_size> packed_right{}, right{};
            std::array<std::uint16_t, vector_size> packed_left{}, left{}, exceptions{}, positions{};
            for (std::size_t i = 0; i < state.right_bit_width * (vector_size / 64); ++i) packed_right[i] = reader.get(8);
            for (std::size_t i = 0; i < state.left_bit_width * (vector_size / 16); ++i) packed_left[i] = static_cast<std::uint16_t>(reader.get(2));
            std::array<bool, vector_size> seen{};
            for (std::uint16_t i = 0; i < exceptions_count; ++i) {
                positions[i] = static_cast<std::uint16_t>(reader.get(2)); exceptions[i] = static_cast<std::uint16_t>(reader.get(2));
                if (positions[i] >= vector_size || seen[positions[i]]) throw std::runtime_error("invalid persisted ALP-RD exception position");
                seen[positions[i]] = true;
            }
            fastlanes::generated::unffor::fallback::scalar::unffor(packed_right.data(), right.data(), state.right_bit_width, &state.right_for_base);
            fastlanes::generated::unffor::fallback::scalar::unffor(packed_left.data(), left.data(), state.left_bit_width, &state.left_for_base);
            for (std::size_t i = 0; i < vector_size; ++i) {
                if (seen[i]) left[i] = 0;
                else if (left[i] >= state.actual_dictionary_size) throw std::runtime_error("persisted ALP-RD dictionary index overflow");
            }
            alp::rd_encoder<double>::decode(result.data() + offset, right.data(), left.data(), exceptions.data(), positions.data(), &exceptions_count, state);
        }
    } else if (scheme == alp::Scheme::ALP) {
        for (std::size_t offset = 0; offset < count; offset += vector_size) {
            const auto width = static_cast<alp::bw_t>(reader.get(1));
            const auto fac = static_cast<std::uint8_t>(reader.get(1)), exp = static_cast<std::uint8_t>(reader.get(1));
            const auto base = std::bit_cast<std::int64_t>(reader.get(8));
            const auto exceptions_count = static_cast<std::uint16_t>(reader.get(2));
            if (width > 64 || fac > alp::Constants<double>::MAX_EXPONENT || exp > alp::Constants<double>::MAX_EXPONENT || exceptions_count > vector_size)
                throw std::runtime_error("invalid persisted ALP vector parameters");
            std::array<std::int64_t, vector_size> packed{}, unpacked{};
            std::array<double, vector_size> exceptions{};
            std::array<std::uint16_t, vector_size> positions{};
            for (std::size_t i = 0; i < width * (vector_size / 64); ++i) packed[i] = std::bit_cast<std::int64_t>(reader.get(8));
            std::array<bool, vector_size> seen{};
            for (std::uint16_t i = 0; i < exceptions_count; ++i) {
                positions[i] = static_cast<std::uint16_t>(reader.get(2)); exceptions[i] = std::bit_cast<double>(reader.get(8));
                if (positions[i] >= vector_size || seen[positions[i]]) throw std::runtime_error("invalid persisted ALP exception position");
                seen[positions[i]] = true;
            }
            fastlanes::generated::unffor::fallback::scalar::unffor(packed.data(), unpacked.data(), width, &base);
            alp::decoder<double>::decode(unpacked.data(), fac, exp, result.data() + offset);
            alp::decoder<double>::patch_exceptions(result.data() + offset, exceptions.data(), positions.data(), &exceptions_count);
        }
    } else throw std::runtime_error("invalid persisted ALP scheme");
    reader.finish();
    return result;
}
} // namespace framed
