#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Gorilla/CompressorGorilla.hpp"
#include "Gorilla/DecompressorGorilla.hpp"
#include "Chimp/CompressorChimp.hpp"
#include "Chimp/DecompressorChimp.hpp"
#include "Chimp128/CompressorChimp128.hpp"
#include "Chimp128/DecompressorChimp128.hpp"
#include "TSXor/CompressorTSXor.hpp"
#include "TSXor/DecompressorTSXor.hpp"
#define NEATS_ADAPTERS_CODEC_HEADERS_INCLUDED 1
#include "rle/neats_adapters.hpp"
#undef NEATS_ADAPTERS_CODEC_HEADERS_INCLUDED
#include "rle/rle_layer.hpp"

namespace fs = std::filesystem;

namespace harness {

class Codec {
public:
    virtual ~Codec() = default;
    virtual std::string name() const = 0;
    virtual std::vector<uint8_t> encode(const double *x, size_t n) = 0;
    virtual std::vector<double> decode(const std::vector<uint8_t> &buf, size_t n) = 0;
    virtual uint64_t encoded_bits(const double *x, size_t n) = 0;
    virtual uint64_t padded_bits(const double *x, size_t n) = 0;
    virtual void prefix_bits(const double *x, size_t n,
                             std::vector<uint64_t> &out) = 0;
    // NeaTS's baseline block counter; Gorilla is read after close(), while
    // Chimp/Chimp128 expose their pre-finalization getSize() counter.
    virtual uint64_t payload_bits(const double *x, size_t n) = 0;
};

enum class Kind : uint8_t {
    Gorilla = 1, Chimp = 2, Chimp128 = 3, TSXor = 4,
    RleGorilla = 5, RleChimp = 6, RleChimp128 = 7, RleTSXor = 8
};

enum class InputMode : uint8_t { IntAsDouble, RawBits, Decimal };

enum class OracleCostModel : uint8_t { Payload, ByteAligned, Current };

// ALP is a vector codec.  The historical oracle allowed a vector stream to
// begin on any point of the fine search grid (the cost table has 64-value
// aligned starts).  For a fairer comparison with a conventional ALP layout we
// also support starts aligned to the series origin at 1024-value boundaries.
enum class AlpAlignment : uint8_t { Grid, Series };

static const char *alp_alignment_name(AlpAlignment alignment) {
    return alignment == AlpAlignment::Series ? "series" : "grid";
}

static const char *oracle_cost_model_name(OracleCostModel model) {
    switch (model) {
        case OracleCostModel::Payload: return "payload";
        case OracleCostModel::ByteAligned: return "byte-aligned";
        default: return "current";
    }
}

static const char *input_mode_name(InputMode mode) {
    switch (mode) {
        case InputMode::RawBits: return "raw_bits";
        case InputMode::Decimal: return "decimal";
        default: return "int_as_double";
    }
}

struct Packed {
    uint64_t stream_bits = 0;
    uint64_t payload_bits = 0;
    uint64_t padded_bits = 0;
    std::vector<uint8_t> bytes;
};

static void put_u64(std::vector<uint8_t> &out, uint64_t v) {
    for (int i = 7; i >= 0; --i) out.push_back(static_cast<uint8_t>(v >> (i * 8)));
}

static uint64_t get_u64(const std::vector<uint8_t> &in, size_t off) {
    if (off + 8 > in.size()) throw std::runtime_error("truncated harness buffer");
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) v = (v << 8) | in[off + i];
    return v;
}

static std::vector<uint8_t> bitstream_bytes(const BitStream &bs) {
    // Preserve the baseline BitStream buckets verbatim. In particular, its
    // final partial bucket is right-aligned; converting it to a conventional
    // bit-packed byte string is easy to get wrong and can corrupt the stream.
    std::vector<uint8_t> out;
    out.reserve(bs.data.size() * sizeof(uint64_t));
    for (const uint64_t word : bs.data) {
        for (int i = 7; i >= 0; --i)
            out.push_back(static_cast<uint8_t>(word >> (i * 8)));
    }
    return out;
}

static BitStream bitstream_from_bytes(const std::vector<uint8_t> &bytes,
                                      uint64_t bits) {
    std::deque<uint64_t> words;
    for (size_t off = 0; off + sizeof(uint64_t) <= bytes.size(); off += sizeof(uint64_t))
        words.push_back(get_u64(bytes, off));
    // The legacy baseline decoders do not bounds-check BitStream::get().
    // Keep a finite zero sentinel after the serialized buckets so malformed
    // control words cannot walk into an empty deque.
    for (size_t i = 0; i < 128; ++i) words.push_back(0);
    if (words.empty()) words.push_back(0);
    return BitStream(words, bits);
}

static std::vector<uint8_t> pack(Kind kind, size_t n, const Packed &p) {
    std::vector<uint8_t> out{'N', 'H', 'C', '1', static_cast<uint8_t>(kind), 0, 0, 0};
    put_u64(out, static_cast<uint64_t>(n));
    put_u64(out, p.stream_bits);
    put_u64(out, p.payload_bits);
    put_u64(out, p.bytes.size());
    out.insert(out.end(), p.bytes.begin(), p.bytes.end());
    return out;
}

static Packed unpack(const std::vector<uint8_t> &buf, Kind expected, size_t n) {
    if (buf.size() < 40 || buf[0] != 'N' || buf[1] != 'H' || buf[2] != 'C' || buf[3] != '1')
        throw std::runtime_error("invalid harness buffer");
    if (static_cast<Kind>(buf[4]) != expected || get_u64(buf, 8) != n)
        throw std::runtime_error("harness buffer metadata mismatch");
    const uint64_t stream_bits = get_u64(buf, 16);
    const uint64_t payload_bits = get_u64(buf, 24);
    const uint64_t nbytes = get_u64(buf, 32);
    if (40 + nbytes != buf.size()) throw std::runtime_error("invalid harness buffer length");
    return {stream_bits, payload_bits, 0,
            std::vector<uint8_t>(buf.begin() + 40, buf.end())};
}

template <typename T>
static void checked_reserve(std::vector<T> &v, size_t n, const char *where) {
    if (n > v.max_size()) {
        throw std::runtime_error(std::string(where) + ": invalid reserve length " +
                                 std::to_string(n) + " (max " +
                                 std::to_string(v.max_size()) + ")");
    }
    v.reserve(n);
}

class BaselineCodec final : public Codec {
    Kind kind_;
    std::string name_;

    static bool is_rle_kind(Kind kind) {
        return kind == Kind::RleGorilla || kind == Kind::RleChimp ||
               kind == Kind::RleChimp128 || kind == Kind::RleTSXor;
    }

    template <class Inner>
    static Packed build_rle(const double *x, size_t n) {
        rle::RleCodec<Inner> codec;
        Packed p;
        p.payload_bits = codec.payload_bits(x, n);
        p.stream_bits = codec.encoded_bits(x, n);
        p.padded_bits = ((p.stream_bits + 7) / 8) * 8;
        p.bytes = codec.encode(x, n);
        return p;
    }

    template <class Inner>
    static std::vector<double> decode_rle(const Packed &p, size_t n) {
        rle::RleCodec<Inner> codec;
        return codec.decode(p.bytes, n);
    }

    template <class Inner>
    static void prefix_rle(const double *x, size_t n,
                           std::vector<uint64_t> &out) {
        rle::RleCodec<Inner> codec;
        codec.prefix_bits(x, n, out);
    }

    Packed build(const double *x, size_t n) const {
        if (n == 0) throw std::invalid_argument("empty streams are not supported by baseline codecs");
        Packed p;
        if (kind_ == Kind::RleGorilla) {
            return build_rle<neats_adapters::Gorilla>(x, n);
        } else if (kind_ == Kind::RleChimp) {
            return build_rle<neats_adapters::Chimp>(x, n);
        } else if (kind_ == Kind::RleChimp128) {
            return build_rle<neats_adapters::Chimp128>(x, n);
        } else if (kind_ == Kind::RleTSXor) {
            return build_rle<neats_adapters::TSXor>(x, n);
        } else if (kind_ == Kind::Gorilla) {
            CompressorGorilla<double> c(x[0]);
            for (size_t i = 1; i < n; ++i) c.addValue(x[i]);
            p.payload_bits = c.bs_values.size();
            c.close();
            p.stream_bits = c.bs_values.size();
            p.padded_bits = ((p.stream_bits + 7) / 8) * 8;
            p.bytes = bitstream_bytes(c.bs_values);
        } else if (kind_ == Kind::Chimp) {
            CompressorChimp<double> c(x[0]);
            for (size_t i = 1; i < n; ++i) c.addValue(x[i]);
            p.payload_bits = c.getSize();
            c.close();
            const auto bs = c.getBuffer();
            p.stream_bits = bs.size();
            p.padded_bits = ((p.stream_bits + 7) / 8) * 8;
            p.bytes = bitstream_bytes(bs);
        } else if (kind_ == Kind::Chimp128) {
            CompressorChimp128<double> c(x[0]);
            for (size_t i = 1; i < n; ++i) c.addValue(x[i]);
            p.payload_bits = c.getSize();
            c.close();
            const auto bs = c.getBuffer();
            p.stream_bits = bs.size();
            p.padded_bits = ((p.stream_bits + 7) / 8) * 8;
            p.bytes = bitstream_bytes(bs);
        } else {
            CompressorTSXor<double> c(x[0]);
            for (size_t i = 1; i < n; ++i) c.addValue(x[i]);
            p.payload_bits = c.getSize();
            c.close();
            p.stream_bits = p.payload_bits;
            p.padded_bits = p.payload_bits;
            p.bytes = c.bytes;
        }
        const uint64_t serialized_bits = static_cast<uint64_t>(p.bytes.size()) * 8;
        if (kind_ == Kind::TSXor) {
            if (p.stream_bits != serialized_bits)
                throw std::runtime_error("TSXor size accounting disagrees with byte buffer");
        } else if (p.stream_bits > serialized_bits) {
            throw std::runtime_error("bitstream logical size exceeds serialized buckets");
        }
        return p;
    }

public:
    BaselineCodec(Kind k, std::string n) : kind_(k), name_(std::move(n)) {}
    std::string name() const override { return name_; }

    std::vector<uint8_t> encode(const double *x, size_t n) override {
        return pack(kind_, n, build(x, n));
    }

    std::vector<double> decode(const std::vector<uint8_t> &buf, size_t n) override {
        const Packed p = unpack(buf, kind_, n);
        if (kind_ == Kind::RleGorilla)
            return decode_rle<neats_adapters::Gorilla>(p, n);
        if (kind_ == Kind::RleChimp)
            return decode_rle<neats_adapters::Chimp>(p, n);
        if (kind_ == Kind::RleChimp128)
            return decode_rle<neats_adapters::Chimp128>(p, n);
        if (kind_ == Kind::RleTSXor)
            return decode_rle<neats_adapters::TSXor>(p, n);
        std::vector<double> out;
        checked_reserve(out, n, "decode");
        if (kind_ == Kind::TSXor) {
            DecompressorTSXor<double> d(p.bytes, n);
            out.push_back(d.storedValue);
            for (size_t i = 1; i < n; ++i) {
                if (!d.hasNext()) throw std::runtime_error("TSXor ended early");
                out.push_back(d.storedValue);
            }
            return out;
        }
        BitStream bs = bitstream_from_bytes(p.bytes, p.stream_bits);
        if (kind_ == Kind::Gorilla) {
            DecompressorGorilla<double> d(bs, n);
            out.push_back(d.storedValue);
            for (size_t i = 1; i < n; ++i) {
                if (!d.hasNext()) throw std::runtime_error("Gorilla ended early");
                out.push_back(d.storedValue);
            }
        } else if (kind_ == Kind::Chimp) {
            DecompressorChimp<double> d(bs, n);
            out.push_back(d.storedValue);
            for (size_t i = 1; i < n; ++i) {
                if (!d.hasNext()) throw std::runtime_error("Chimp ended early");
                out.push_back(d.storedValue);
            }
        } else {
            DecompressorChimp128<double> d(bs, n);
            out.push_back(d.storedValue);
            for (size_t i = 1; i < n; ++i) {
                if (!d.hasNext()) throw std::runtime_error("Chimp128 ended early");
                out.push_back(d.storedValue);
            }
        }
        return out;
    }

    uint64_t encoded_bits(const double *x, size_t n) override { return build(x, n).stream_bits; }
    uint64_t format_overhead_bits(uint64_t rle_header_bits) const {
        return is_rle_kind(kind_) ? rle_header_bits : 0;
    }
    uint64_t oracle_cost_bits(const double *x, size_t n,
                              OracleCostModel model,
                              uint64_t rle_header_bits) {
        // Packed::payload_bits is captured before close(); stream_bits is the
        // legacy closed stream used only by the conservative current model.
        const Packed p = build(x, n);
        uint64_t bits = p.payload_bits;
        if (model == OracleCostModel::ByteAligned) {
            if (bits > std::numeric_limits<uint64_t>::max() - 7)
                throw std::overflow_error("oracle byte-alignment overflow");
            bits = ((bits + 7) / 8) * 8;
        } else if (model == OracleCostModel::Current) {
            bits = p.stream_bits;
            const uint64_t extra = format_overhead_bits(rle_header_bits);
            if (bits > std::numeric_limits<uint64_t>::max() - extra)
                throw std::overflow_error("codec format overhead overflow");
            bits += extra;
        }
        return bits;
    }
    uint64_t finalization_overhead_bits() const {
        // Gorilla/Chimp/Chimp128 append a 64-bit trailer before BitStream's
        // 133-bit close marker. TSXor is already byte-oriented and has no
        // separate finalization bits.
        return kind_ == Kind::TSXor || kind_ == Kind::RleTSXor ? 0 : 197;
    }
    uint64_t padded_bits(const double *x, size_t n) override { return build(x, n).padded_bits; }
    uint64_t payload_bits(const double *x, size_t n) override {
        const Packed p = build(x, n);
        // NeaTS's original Gorilla wrapper reads bs_values.size() after
        // close(), so its block total includes Gorilla finalization bits.
        // Chimp/Chimp128 getSize() remains the pre-finalization counter, and
        // TSXor is already byte-oriented.
        if (kind_ == Kind::Gorilla) return p.stream_bits;
        return p.payload_bits;
    }

    void prefix_bits(const double *x, size_t n, std::vector<uint64_t> &out) override {
        out.clear();
        if (n == 0) return;
        checked_reserve(out, n, "prefix_bits");
        if (kind_ == Kind::RleGorilla) {
            prefix_rle<neats_adapters::Gorilla>(x, n, out);
        } else if (kind_ == Kind::RleChimp) {
            prefix_rle<neats_adapters::Chimp>(x, n, out);
        } else if (kind_ == Kind::RleChimp128) {
            prefix_rle<neats_adapters::Chimp128>(x, n, out);
        } else if (kind_ == Kind::RleTSXor) {
            prefix_rle<neats_adapters::TSXor>(x, n, out);
        } else if (kind_ == Kind::Gorilla) {
            CompressorGorilla<double> c(x[0]); out.push_back(c.bs_values.size());
            for (size_t i = 1; i < n; ++i) { c.addValue(x[i]); out.push_back(c.bs_values.size()); }
        } else if (kind_ == Kind::Chimp) {
            CompressorChimp<double> c(x[0]); out.push_back(c.getSize());
            for (size_t i = 1; i < n; ++i) { c.addValue(x[i]); out.push_back(c.getSize()); }
        } else if (kind_ == Kind::Chimp128) {
            CompressorChimp128<double> c(x[0]); out.push_back(c.getSize());
            for (size_t i = 1; i < n; ++i) { c.addValue(x[i]); out.push_back(c.getSize()); }
        } else {
            CompressorTSXor<double> c(x[0]); out.push_back(c.getSize());
            for (size_t i = 1; i < n; ++i) { c.addValue(x[i]); out.push_back(c.getSize()); }
        }
    }
};

// ALP is a vector codec rather than a streaming codec.  The standalone
// AlpCostTable tool records the declared ALP/ALP-RD payload cost for each
// (start,length) candidate and verifies a canonical bit-exact serialization.
// The oracle consumes that table without linking the optional ALP library into
// the normal harness executable.
struct AlpCostTable {
    std::map<std::pair<size_t, size_t>, uint64_t> official_bits;
    size_t vector_size = 1024;
    size_t start_stride = 0;
    size_t max_candidate_len = 0;

    static std::vector<std::string> split_csv(const std::string &line) {
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) fields.push_back(field);
        return fields;
    }

    static AlpCostTable load(const fs::path &path, bool allow_empty = false) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open ALP cost table: " + path.string());
        std::string line;
        if (!std::getline(in, line) ||
            line != "start,length,vectors,official_payload_bits,canonical_bits,canonical_padded_bits,official_bits_per_value,canonical_bits_per_value,roundtrip_ok") {
            throw std::runtime_error("invalid ALP cost-table header: " + path.string());
        }
        AlpCostTable table;
        size_t row = 1;
        while (std::getline(in, line)) {
            ++row;
            if (line.empty()) continue;
            const auto fields = split_csv(line);
            if (fields.size() < 9)
                throw std::runtime_error("invalid ALP cost-table row " + std::to_string(row));
            const size_t start = std::stoull(fields[0]);
            const size_t length = std::stoull(fields[1]);
            const uint64_t bits = std::stoull(fields[3]);
            const int roundtrip_ok = std::stoi(fields[8]);
            if (length == 0 || length % table.vector_size != 0 || roundtrip_ok != 1)
                throw std::runtime_error("invalid or unverified ALP row " + std::to_string(row));
            // Requested frozen segments may begin at any raw-value offset.
            // The oracle enforces its own search grid when looking up edges.
            const auto key = std::make_pair(start, length);
            if (!table.official_bits.emplace(key, bits).second)
                throw std::runtime_error("duplicate ALP cost-table candidate at row " +
                                         std::to_string(row));
            table.max_candidate_len = std::max(table.max_candidate_len, length);
            if (table.start_stride == 0 || start != 0)
                table.start_stride = table.start_stride == 0 ? start : std::gcd(table.start_stride, start);
        }
        if (in.bad()) throw std::runtime_error("failed reading ALP cost table: " + path.string());
        if (table.official_bits.empty() && !allow_empty)
            throw std::runtime_error("ALP cost table has no candidates: " + path.string());
        // The first non-zero start may be absent from a tiny table.  The
        // oracle still validates every requested start through exact lookup.
        return table;
    }

    bool lookup(size_t start, size_t length, uint64_t &bits) const {
        const auto it = official_bits.find({start, length});
        if (it == official_bits.end()) return false;
        bits = it->second;
        return true;
    }

    bool largest_prefix(size_t start, size_t max_length,
                        size_t &length, uint64_t &bits) const {
        auto it = official_bits.lower_bound({start, 0});
        bool found = false;
        for (; it != official_bits.end() && it->first.first == start; ++it) {
            if (it->first.second > max_length) break;
            if (!found || it->first.second > length) {
                length = it->first.second;
                bits = it->second;
                found = true;
            }
        }
        return found;
    }
};

class OracleCodec {
    std::optional<BaselineCodec> baseline_;
    std::shared_ptr<const AlpCostTable> alp_;
    const double *series_base_ = nullptr;
    std::string name_;

    size_t alp_start(const double *x) const {
        if (!series_base_ || x < series_base_)
            return std::numeric_limits<size_t>::max();
        return static_cast<size_t>(x - series_base_);
    }

public:
    explicit OracleCodec(BaselineCodec codec)
        : baseline_(std::move(codec)), name_(baseline_->name()) {}

    explicit OracleCodec(std::shared_ptr<const AlpCostTable> table)
        : alp_(std::move(table)), name_("alp") {}

    std::string name() const { return name_; }
    bool is_alp() const { return static_cast<bool>(alp_); }
    void set_series_base(const double *base) { series_base_ = base; }

    uint64_t oracle_cost_bits(const double *x, size_t n,
                              OracleCostModel model, uint64_t rle_header_bits) {
        if (baseline_) return baseline_->oracle_cost_bits(x, n, model, rle_header_bits);
        const size_t start = alp_start(x);
        uint64_t bits = 0;
        if (start == std::numeric_limits<size_t>::max() || !alp_->lookup(start, n, bits))
            return std::numeric_limits<uint64_t>::max();
        if (model == OracleCostModel::ByteAligned) {
            if (bits > std::numeric_limits<uint64_t>::max() - 7)
                throw std::overflow_error("ALP byte-alignment overflow");
            bits = ((bits + 7) / 8) * 8;
        }
        // ALP's official payload is already a closed vector cost.  There is
        // no legacy close marker and no RLE transport header to add.
        return bits;
    }

    bool largest_alp_prefix(const double *x, size_t max_length,
                            OracleCostModel model, size_t &length,
                            uint64_t &bits) const {
        if (!is_alp()) return false;
        const size_t start = alp_start(x);
        if (start == std::numeric_limits<size_t>::max()) return false;
        if (!alp_->largest_prefix(start, max_length, length, bits)) return false;
        if (model == OracleCostModel::ByteAligned) {
            if (bits > std::numeric_limits<uint64_t>::max() - 7)
                throw std::overflow_error("ALP byte-alignment overflow");
            bits = ((bits + 7) / 8) * 8;
        }
        return true;
    }

    uint64_t finalization_overhead_bits() const {
        return is_alp() ? 0 : baseline_->finalization_overhead_bits();
    }

    uint64_t format_overhead_bits(uint64_t rle_header_bits) const {
        return is_alp() ? 0 : baseline_->format_overhead_bits(rle_header_bits);
    }

    void prefix_bits(const double *x, size_t n, std::vector<uint64_t> &out) {
        if (is_alp()) throw std::runtime_error("ALP has no streaming prefix counter");
        baseline_->prefix_bits(x, n, out);
    }

    bool verified_roundtrip(const double *x, size_t n) {
        // ALP table loading already requires the standalone tool's canonical
        // bit-exact round-trip check to have succeeded for every entry.
        if (is_alp()) return true;
        const auto decoded = baseline_->decode(baseline_->encode(x, n), n);
        if (decoded.size() != n) return false;
        for (size_t i = 0; i < n; ++i)
            if (std::bit_cast<uint64_t>(decoded[i]) != std::bit_cast<uint64_t>(x[i]))
                return false;
        return true;
    }
};

static uint64_t bits(double x) { return std::bit_cast<uint64_t>(x); }

static bool same_bits(const std::vector<double> &a, const std::vector<double> &b,
                     size_t &bad) {
    if (a.size() != b.size()) { bad = std::min(a.size(), b.size()); return false; }
    for (size_t i = 0; i < a.size(); ++i) if (bits(a[i]) != bits(b[i])) { bad = i; return false; }
    return true;
}

// .double_bin is the headerless float64 suffix used by the SCADA inputs.
// Keep the logical series identity equal to the filename stem for either
// suffix; never silently substitute one raw input when both names exist.
static bool is_series_file(const fs::path &path) {
    return path.extension() == ".bin" || path.extension() == ".double_bin";
}

static fs::path resolve_series_file(const fs::path &data_dir,
                                    const std::string &series) {
    const fs::path bin = data_dir / (series + ".bin");
    const fs::path double_bin = data_dir / (series + ".double_bin");
    const bool has_bin = fs::is_regular_file(bin);
    const bool has_double_bin = fs::is_regular_file(double_bin);
    if (has_bin && has_double_bin)
        throw std::runtime_error("ambiguous series input: both " + bin.string() +
                                 " and " + double_bin.string() + " exist");
    if (has_bin) return bin;
    if (has_double_bin) return double_bin;
    throw std::runtime_error("no .bin or .double_bin input for series " +
                             series + " in " + data_dir.string());
}

static std::vector<fs::path> discover_series_files(
        const fs::path &data_dir, const std::string &series_filter) {
    if (!series_filter.empty()) return {resolve_series_file(data_dir, series_filter)};
    std::map<std::string, fs::path> unique;
    for (const auto &entry : fs::directory_iterator(data_dir)) {
        if (!entry.is_regular_file() || !is_series_file(entry.path())) continue;
        const auto [it, inserted] = unique.emplace(entry.path().stem().string(), entry.path());
        if (!inserted)
            throw std::runtime_error("ambiguous series input: both " + it->second.string() +
                                     " and " + entry.path().string() + " exist");
    }
    std::vector<fs::path> files;
    for (const auto &[series, path] : unique) files.push_back(path);
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("no .bin or .double_bin datasets in " + data_dir.string());
    return files;
}

static std::vector<double> read_series(const fs::path &path, size_t max_values,
                                       InputMode mode, size_t decimal_digits) {
    const uint64_t bytes = fs::file_size(path);
    if (bytes % 8 != 0) throw std::runtime_error("dataset is not 64-bit aligned: " + path.string());
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    uint64_t first = 0;
    in.read(reinterpret_cast<char *>(&first), sizeof(first));
    const bool has_count = first == (bytes - 8) / 8 && first != 0;
    const uint64_t total = has_count ? first : bytes / 8;
    const size_t n = static_cast<size_t>(std::min<uint64_t>(total, max_values ? max_values : total));
    std::vector<double> out(n);
    in.clear();
    in.seekg(has_count ? 8 : 0);
    if (mode == InputMode::Decimal && decimal_digits > 22)
        throw std::invalid_argument("decimal digits must be <= 22");
    double scale = 1.0;
    for (size_t i = 0; i < decimal_digits; ++i) scale *= 10.0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t word = 0;
        in.read(reinterpret_cast<char *>(&word), sizeof(word));
        if (!in) throw std::runtime_error("short dataset: " + path.string());
        if (mode == InputMode::RawBits)
            out[i] = std::bit_cast<double>(word);
        else {
            const int64_t value = std::bit_cast<int64_t>(word);
            if (mode == InputMode::Decimal &&
                (value > 9007199254740991LL || value < -9007199254740991LL))
                throw std::runtime_error("decimal variant requires int64 values below 2^53");
            out[i] = mode == InputMode::Decimal
                ? static_cast<double>(value) / scale
                : static_cast<double>(value);
        }
    }
    return out;
}

struct Row {
    std::string dataset, input_variant, series, codec, mode;
    size_t n = 0;
    uint64_t total_bits = 0, padded_bits = 0, overhead = 0;
    bool roundtrip = false, prefix_ok = false, matches_driver = false;
};

struct Segment {
    size_t start = 0;
    size_t len = 0;
};

// Table III reference values supplied by the user.  These are compression
// ratios in percent: compressed bits / (64 * n) * 100.  The table's fixed
// block convention matches the harness's block1000 rows.
static double published_ratio_pct(const std::string &series,
                                  const std::string &codec) {
    static const std::map<std::pair<std::string, std::string>, double> refs = {
        {{"IT", "chimp128"}, 29.43}, {{"IT", "chimp"}, 72.30}, {{"IT", "tsxor"}, 30.76}, {{"IT", "gorilla"}, 78.60},
        {{"US", "chimp128"}, 18.94}, {{"US", "chimp"}, 54.55}, {{"US", "tsxor"}, 18.89}, {{"US", "gorilla"}, 57.54},
        {{"ECG", "chimp128"}, 54.11}, {{"ECG", "chimp"}, 43.18}, {{"ECG", "tsxor"}, 20.03}, {{"ECG", "gorilla"}, 45.26},
        {{"WD", "chimp128"}, 43.38}, {{"WD", "chimp"}, 84.09}, {{"WD", "tsxor"}, 46.42}, {{"WD", "gorilla"}, 91.02},
        {{"AP", "chimp128"}, 30.00}, {{"AP", "chimp"}, 35.76}, {{"AP", "tsxor"}, 34.78}, {{"AP", "gorilla"}, 37.67},
        {{"UK", "chimp128"}, 23.13}, {{"UK", "chimp"}, 46.95}, {{"UK", "tsxor"}, 15.85}, {{"UK", "gorilla"}, 53.92},
        {{"GE", "chimp128"}, 21.08}, {{"GE", "chimp"}, 66.90}, {{"GE", "tsxor"}, 21.44}, {{"GE", "gorilla"}, 71.49},
        {{"LON", "chimp128"}, 58.64}, {{"LON", "chimp"}, 61.70}, {{"LON", "tsxor"}, 71.64}, {{"LON", "gorilla"}, 63.09},
        {{"LAT", "chimp128"}, 58.09}, {{"LAT", "chimp"}, 61.44}, {{"LAT", "tsxor"}, 71.93}, {{"LAT", "gorilla"}, 65.02},
        {{"DP", "chimp128"}, 49.53}, {{"DP", "chimp"}, 77.17}, {{"DP", "tsxor"}, 60.91}, {{"DP", "gorilla"}, 83.53},
        {{"CT", "chimp128"}, 36.09}, {{"CT", "chimp"}, 73.25}, {{"CT", "tsxor"}, 30.96}, {{"CT", "gorilla"}, 87.11},
        {{"DU", "chimp128"}, 21.68}, {{"DU", "chimp"}, 39.74}, {{"DU", "tsxor"}, 18.31}, {{"DU", "gorilla"}, 44.49},
        {{"BT", "chimp128"}, 46.90}, {{"BT", "chimp"}, 84.01}, {{"BT", "tsxor"}, 53.88}, {{"BT", "gorilla"}, 92.88},
        {{"BW", "chimp128"}, 71.27}, {{"BW", "chimp"}, 87.16}, {{"BW", "tsxor"}, 82.32}, {{"BW", "gorilla"}, 99.72},
        {{"BM", "chimp128"}, 40.96}, {{"BM", "chimp"}, 61.96}, {{"BM", "tsxor"}, 48.18}, {{"BM", "gorilla"}, 74.67},
        {{"BP", "chimp128"}, 72.09}, {{"BP", "chimp"}, 67.84}, {{"BP", "tsxor"}, 87.86}, {{"BP", "gorilla"}, 82.72}
    };
    const auto it = refs.find({series, codec});
    return it == refs.end() ? std::numeric_limits<double>::quiet_NaN() : it->second;
}

static void write_csv(const fs::path &path, const std::vector<Row> &rows) {
    std::ofstream out(path);
    out << "dataset,input_variant,series,n,codec,mode,total_bits,padded_bits,bits_per_value,ratio_orig_over_comp,"
           "ratio_comp_over_orig_percent,roundtrip_ok,matches_neats_driver,deviation_vs_published,"
           "prefix_checks_ok,finalize_overhead_bits\n";
    for (const auto &r : rows) {
        const double bpv = r.n ? double(r.total_bits) / r.n : 0.0;
        const double ratio = r.total_bits ? (64.0 * r.n) / r.total_bits : 0.0;
        const double pct = r.n ? (100.0 * r.total_bits) / (64.0 * r.n) : 0.0;
        const double reference = r.mode == "block1000"
            ? published_ratio_pct(r.series, r.codec)
            : std::numeric_limits<double>::quiet_NaN();
        const double deviation = std::isfinite(reference) && reference != 0.0
            ? 100.0 * (pct - reference) / reference
            : std::numeric_limits<double>::quiet_NaN();
        out << r.dataset << ',' << r.input_variant << ',' << r.series << ',' << r.n << ',' << r.codec << ',' << r.mode << ','
            << r.total_bits << ',' << r.padded_bits << ',' << std::setprecision(12) << bpv << ',' << ratio << ',' << pct
            << ',' << (r.roundtrip ? 1 : 0) << ',' << (r.matches_driver ? 1 : 0)
            << ',';
        if (std::isfinite(deviation)) out << deviation;
        out << ',' << (r.prefix_ok ? 1 : 0) << ',' << r.overhead << '\n';
    }
}

static std::vector<BaselineCodec> codecs(const std::vector<std::string> &order = {}) {
    static const std::map<std::string, Kind> kinds = {
        {"gorilla", Kind::Gorilla}, {"chimp", Kind::Chimp},
        {"chimp128", Kind::Chimp128}, {"tsxor", Kind::TSXor},
        {"rle+gorilla", Kind::RleGorilla}, {"rle+chimp", Kind::RleChimp},
        {"rle+chimp128", Kind::RleChimp128}, {"rle+tsxor", Kind::RleTSXor}
    };
    const std::vector<std::string> default_order = {"gorilla", "chimp", "chimp128", "tsxor"};
    const auto &selected = order.empty() ? default_order : order;
    std::vector<BaselineCodec> result;
    std::map<std::string, bool> seen;
    for (const auto &name : selected) {
        const auto it = kinds.find(name);
        if (it == kinds.end()) throw std::invalid_argument("unknown codec in --codec-order: " + name);
        if (seen[name]) throw std::invalid_argument("duplicate codec in --codec-order: " + name);
        seen[name] = true;
        result.emplace_back(it->second, name);
    }
    return result;
}

static std::vector<std::string> edge_case_self_test(bool unsafe_gorilla,
                                                    const std::string &codec_filter,
                                                    const std::vector<std::string> &codec_order) {
    const std::vector<double> x = {
        0.0, -0.0, std::bit_cast<double>(uint64_t{0x7ff8000000000001ULL}),
        std::bit_cast<double>(uint64_t{0xfff8000000000042ULL}), 1.0, -1.0, 1.0, 1.0};
    std::vector<std::string> failures;
    for (auto &codec : codecs(codec_order)) {
        if (!codec_filter.empty() && codec.name() != codec_filter) continue;
        if ((codec.name() == "gorilla" || codec.name() == "rle+gorilla") && !unsafe_gorilla) {
            // Gorilla's checked-in decoder is known to misread the NaN control
            // sequence and may consume beyond its logical stream. The RLE
            // wrapper cannot repair an unsafe inner decoder. The regular
            // dataset run still exercises both codecs; skip only this
            // synthetic decode so the marker limitation is explicit.
            failures.push_back(codec.name() + " NaN edge case skipped (baseline marker semantics)");
            continue;
        }
        try {
            const auto decoded = codec.decode(codec.encode(x.data(), x.size()), x.size());
            size_t bad = 0;
            if (!same_bits(x, decoded, bad))
                failures.push_back(codec.name() + " at index " + std::to_string(bad));
        } catch (const std::exception &e) {
            failures.push_back(codec.name() + " (exception: " + e.what() + ")");
        }
    }
    return failures;
}

static int run(const fs::path &data_dir, size_t max_values, size_t random_segments,
               uint64_t seed, const std::string &codec_filter,
               const std::string &series_filter, bool unsafe_gorilla,
               const fs::path &results_path, const fs::path &report_path,
               bool run_edge_tests, InputMode input_mode, size_t decimal_digits,
               const std::vector<std::string> &codec_order) {
    const auto edge_failures = run_edge_tests
        ? edge_case_self_test(unsafe_gorilla, codec_filter, codec_order)
        : std::vector<std::string>{"not run (--no-edge-test)"};
    for (const auto &failure : edge_failures)
        std::cerr << "edge-case check: " << failure << '\n';
    const auto files = discover_series_files(data_dir, series_filter);

    const std::string dataset = data_dir.filename().string();
    std::vector<Row> rows;
    std::vector<std::string> read_failures;
    for (size_t file_idx = 0; file_idx < files.size(); ++file_idx) {
        const auto &file = files[file_idx];
        std::vector<double> x;
        try {
            x = read_series(file, max_values, input_mode, decimal_digits);
        } catch (const std::exception &e) {
            const std::string failure = file.filename().string() + ": " + e.what();
            read_failures.push_back(failure);
            std::cerr << "dataset read exception: " << failure << '\n';
            continue;
        }
        const size_t n = x.size();
        if (n == 0) continue;
        std::cerr << "validating " << file.filename().string() << " (n=" << n << ")\n";

        // Generate the segment oracle once per series.  Use floor(log2(max_len));
        // using ceil(log2(max_len)) makes lo > hi in the final truncated octave
        // and wraps hi - lo + 1 to a huge size_t.
        std::vector<Segment> segments;
        segments.reserve(random_segments);
        std::mt19937_64 seg_rng(seed +
            0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(file_idx + 1));
        for (size_t q = 0; q < random_segments; ++q) {
            const size_t start = static_cast<size_t>(seg_rng() % n);
            const size_t max_len = std::min<size_t>(n - start, 65536);
            const size_t max_exp = std::bit_width(max_len) - 1;
            const size_t exp = static_cast<size_t>(seg_rng() % (max_exp + 1));
            const size_t lo = size_t{1} << exp;
            const size_t hi = exp == max_exp ? max_len : (size_t{1} << (exp + 1)) - 1;
            if (lo > hi || hi > max_len)
                throw std::logic_error("segment sampler invariant violated");
            const size_t len = lo + static_cast<size_t>(seg_rng() % (hi - lo + 1));
            if (len > max_len || start + len > n)
                throw std::logic_error("segment sampler produced out-of-range segment");
            segments.push_back({start, len});
        }

        for (auto &codec : codecs(codec_order)) {
            if (!codec_filter.empty() && codec.name() != codec_filter) continue;
            try {
            const bool decoder_safe = true;
            // Compute prefix costs before invoking any baseline decoder.  The
            // legacy decoders are not bounds-safe; keeping this independent
            // check first makes any decoder-side stack/heap corruption
            // immediately distinguishable from an encoder/prefix failure.
            std::vector<uint64_t> prefix;
            codec.prefix_bits(x.data(), n, prefix);
            bool prefix_ok = prefix.size() == n;
            for (size_t i = 1; i < prefix.size(); ++i) prefix_ok &= prefix[i] >= prefix[i - 1];
            const uint64_t total_bits = codec.encoded_bits(x.data(), n);
            if (prefix.empty() || prefix.back() > total_bits ||
                total_bits > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                throw std::runtime_error("invalid finalization accounting");
            const int64_t signed_overhead = static_cast<int64_t>(total_bits - prefix.back());
            if (signed_overhead < 0 || signed_overhead > 4096)
                throw std::runtime_error("finalization overhead is not small/non-negative: " +
                                         std::to_string(signed_overhead));
            const uint64_t overhead = static_cast<uint64_t>(signed_overhead);

            const auto full_buf = codec.encode(x.data(), n);
            std::vector<double> decoded;
            if (decoder_safe) decoded = codec.decode(full_buf, n);
            size_t bad = 0;
            const bool full_ok = decoder_safe && same_bits(x, decoded, bad);
            if (!full_ok) {
                std::cerr << "roundtrip failure: " << file.filename().string() << "/"
                          << codec.name() << "/full";
                if (decoder_safe) {
                    std::cerr << " index " << bad;
                    if (codec.name() == "gorilla" && bad < n) {
                        const uint64_t prev = bad ? bits(x[bad - 1]) : 0;
                        const uint64_t curr = bits(x[bad]);
                        const uint64_t xor_value = prev ^ curr;
                        if (xor_value != 0)
                            std::cerr << " xor_lz=" << std::countl_zero(xor_value)
                                      << " xor_tz=" << std::countr_zero(xor_value);
                    }
                }
                else std::cerr << " (decoder skipped: unsafe baseline)";
                std::cerr << '\n';
            }
            for (const Segment &segment : segments) {
                const size_t start = segment.start;
                const size_t len = segment.len;
                std::vector<uint64_t> seg_prefix;
                codec.prefix_bits(x.data() + start, len, seg_prefix);
                prefix_ok &= codec.encoded_bits(x.data() + start, len) == seg_prefix.back() + overhead;
                if (decoder_safe) {
                    const auto seg_buf = codec.encode(x.data() + start, len);
                    const auto seg_dec = codec.decode(seg_buf, len);
                    size_t seg_bad = 0;
                    prefix_ok &= same_bits(std::vector<double>(x.begin() + start, x.begin() + start + len), seg_dec, seg_bad);
                }
            }
            if (!prefix_ok)
                std::cerr << "prefix check failure: " << file.filename().string() << "/"
                          << codec.name() << '\n';
            rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(), codec.name(), "full",
                            n, codec.encoded_bits(x.data(), n), codec.padded_bits(x.data(), n), overhead,
                            full_ok, prefix_ok, false});

            bool block_ok = true;
            uint64_t block_bits = 0, block_padded_bits = 0;
            for (size_t start = 0; start < n; start += 1000) {
                const size_t len = std::min<size_t>(1000, n - start);
                const auto b = codec.encode(x.data() + start, len);
                size_t block_bad = 0;
                if (decoder_safe) {
                    const auto d = codec.decode(b, len);
                    block_ok &= same_bits(std::vector<double>(x.begin() + start, x.begin() + start + len), d, block_bad);
                } else {
                    block_ok = false;
                }
                block_bits += codec.payload_bits(x.data() + start, len);
                block_padded_bits += codec.padded_bits(x.data() + start, len);
            }
            rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(), codec.name(), "block1000",
                            n, block_bits, block_padded_bits, overhead, block_ok, prefix_ok, false});
            if (!block_ok)
                std::cerr << "roundtrip failure: " << file.filename().string() << "/"
                          << codec.name() << "/block1000\n";
            } catch (const std::exception &e) {
                std::cerr << "validation exception: " << file.filename().string() << "/"
                          << codec.name() << ": " << e.what() << '\n';
            }
        }
    }
    write_csv(results_path, rows);
    std::ofstream report(report_path);
    report << "# Validation\n\n";
    report << "- Dataset directory: `" << data_dir.string() << "`\n";
    report << "- Input variant: " << input_mode_name(input_mode) << "\n";
    if (input_mode == InputMode::Decimal)
        report << "- Decimal digits: " << decimal_digits << "\n";
    report << "- RNG seed: " << seed << "\n";
    if (!codec_filter.empty()) report << "- Codec filter: " << codec_filter << "\n";
    if (!codec_order.empty()) {
        report << "- Codec order: ";
        for (size_t i = 0; i < codec_order.size(); ++i) {
            if (i) report << ",";
            report << codec_order[i];
        }
        report << "\n";
    }
    if (!series_filter.empty()) report << "- Series filter: " << series_filter << "\n";
    report << "- Unsafe Gorilla mode (synthetic NaN probe only): " << (unsafe_gorilla ? "enabled" : "disabled") << "\n";
    report << "- Prefix segment checks per series/codec: " << random_segments << "\n";
    report << "- Segment oracle: deterministic per-series log-uniform lengths with floor(log2(max_len)); shared by all codecs.\n";
    report << "- Block mode: independent blocks of 1000 values; size is the baseline payload counter.\n";
    report << "- Full mode: encoded stream size includes the codec finalization marker/padding.\n";
    report << "- `encoded_bits` remains the codec's logical bit count; serialized bucket storage is reported separately as `padded_bits`.\n";
    report << "- Gorilla is included in normal data validation; `--unsafe-gorilla` only enables its synthetic NaN probe.\n";
    report << "- Source files use `.bin` or `.double_bin`; the selected input variant controls conversion/reinterpretation before baseline compression.\n\n";
    report << "Dataset read failures:\n";
    if (read_failures.empty()) report << "- none\n\n";
    else {
        for (const auto &failure : read_failures) report << "- " << failure << "\n";
        report << "\n";
    }
    report << "Edge-case bit-pattern self-test (`+0/-0` and NaNs):\n";
    if (!run_edge_tests) report << "- NOT RUN\n\n";
    else if (edge_failures.empty()) report << "- PASS\n\n";
    else {
        for (const auto &failure : edge_failures) report << "- NOTE: " << failure << "\n";
        report << "\n";
    }
    std::map<std::string, std::pair<uint64_t, uint64_t>> overhead_ranges;
    for (const auto &r : rows) {
        if (r.mode != "full") continue;
        auto it = overhead_ranges.find(r.codec);
        if (it == overhead_ranges.end()) overhead_ranges.emplace(r.codec, std::make_pair(r.overhead, r.overhead));
        else { it->second.first = std::min(it->second.first, r.overhead); it->second.second = std::max(it->second.second, r.overhead); }
    }
    report << "Finalization overhead ranges (full-series rows):\n";
    for (const auto &[codec, range] : overhead_ranges)
        report << "- " << codec << ": " << range.first << ".." << range.second << " bits\n";
    report << "\n";
    size_t published_compared = 0;
    size_t published_outliers = 0;
    for (const auto &r : rows) {
        if (r.mode != "block1000" || r.n == 0) continue;
        const double reference = published_ratio_pct(r.series, r.codec);
        if (!std::isfinite(reference) || reference == 0.0) continue;
        const double actual = 100.0 * double(r.total_bits) / (64.0 * double(r.n));
        const double deviation = 100.0 * (actual - reference) / reference;
        ++published_compared;
        if (std::abs(deviation) > 3.0) ++published_outliers;
    }
    report << "Published reference source: user-supplied Table III image; values are compression ratio (%) = compressed/original * 100.\n";
    report << "Published comparisons use block1000 rows: " << published_compared
           << " rows, " << published_outliers << " with absolute relative deviation > 3%.\n";
    report << "BT Gorilla note: the corrected lossless encoder retains a full-width XOR window after a zero-leading/zero-trailing sign change. The previous shortcut cleared one XOR bit and could report a smaller but lossy size; it must not be used as the reference result.\n";
    report << "Chimp note: a small systematic offset versus Table III should be checked against the unchanged NeaTS wrapper before attributing it to this harness.\n";
    report << "The external NeaTS `Benchmark` executable was not run by this harness, so `matches_neats_driver` is 0 rather than an unverified claim. The block calculation does use the same source-level wrapper formula.\n\n";
    report << "See `results.csv` for per-series results.\n";
    return 0;
}

struct OracleRow {
    std::string dataset, input_variant, series, strategy, codec;
    size_t n = 0, block_len = 0, segments = 0, covered = 0;
    uint64_t total_bits = 0;
};

static std::vector<size_t> oracle_lengths(size_t min_len, size_t max_len,
                                          bool every_multiple) {
    if (min_len == 0 || max_len < min_len)
        throw std::invalid_argument("oracle length range is invalid");
    std::vector<size_t> lengths;
    if (every_multiple) {
        for (size_t length = min_len;;) {
            lengths.push_back(length);
            if (length > max_len - min_len) break;
            length += min_len;
        }
        return lengths;
    }
    for (size_t length = min_len;;) {
        lengths.push_back(length);
        if (length > max_len / 2) break;
        length *= 2;
    }
    return lengths;
}

static void write_oracle_csv(const fs::path &path, const std::vector<OracleRow> &rows) {
    std::ofstream out(path);
    out << "dataset,input_variant,series,n,strategy,block_len,codec,total_bits,bits_per_value,segments,covered_values\n";
    for (const auto &r : rows) {
        const double bpv = r.n ? double(r.total_bits) / double(r.n) : 0.0;
        out << r.dataset << ',' << r.input_variant << ',' << r.series << ',' << r.n << ','
            << r.strategy << ',' << r.block_len << ',' << r.codec << ',' << r.total_bits << ','
            << std::setprecision(12) << bpv << ',' << r.segments << ',' << r.covered << '\n';
    }
}

struct PlanSegment {
    size_t start = 0;
    size_t length = 0;
    std::string codec;
};

// Measure only explicitly requested frozen intervals. This mode performs no
// boundary search and is independent of the oracle candidate grid. Costs use
// exactly the same cold-start definition as score_plan below.
static int write_requested_segment_costs(
        const fs::path &data_dir, const std::string &series, size_t max_values,
        InputMode input_mode, size_t decimal_digits, OracleCostModel cost_model,
        size_t metadata_bits, uint64_t rle_header_bits,
        const std::vector<std::string> &codec_order,
        const fs::path &alp_cost_table_path, const fs::path &request_path,
        const fs::path &output_path) {
    if (series.empty() || request_path.empty() || output_path.empty() || codec_order.empty())
        throw std::invalid_argument("segment measurement requires --series, --codec-order, --segment-requests and --segment-costs");
    const auto x = read_series(resolve_series_file(data_dir, series), max_values,
                               input_mode, decimal_digits);
    if (x.empty()) throw std::runtime_error("empty segment-measurement series");
    std::vector<std::string> streaming;
    bool wants_alp = false;
    for (const auto &name : codec_order) {
        if (name == "alp") wants_alp = true;
        else streaming.push_back(name);
    }
    std::vector<OracleCodec> selected;
    if (!streaming.empty())
        for (auto &codec : codecs(streaming)) selected.emplace_back(std::move(codec));
    if (wants_alp) {
        if (alp_cost_table_path.empty())
            throw std::invalid_argument("ALP segment requests require --plan-alp-cost-table");
        // A frozen partition can contain only short streaming regions.  Its
        // targeted ALP request file then legitimately produces a header-only
        // table. Actual ALP requests still require an exact lookup below.
        selected.emplace_back(std::make_shared<const AlpCostTable>(AlpCostTable::load(alp_cost_table_path, true)));
    }
    std::map<std::string, size_t> codec_index;
    for (size_t i = 0; i < selected.size(); ++i) {
        selected[i].set_series_base(x.data());
        codec_index.emplace(selected[i].name(), i);
    }
    std::ifstream in(request_path);
    if (!in) throw std::runtime_error("cannot open segment requests: " + request_path.string());
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty segment requests");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = AlpCostTable::split_csv(line);
    std::map<std::string, size_t> column;
    for (size_t i = 0; i < header.size(); ++i)
        if (!column.emplace(header[i], i).second)
            throw std::runtime_error("duplicate segment request column: " + header[i]);
    for (const std::string name : {"dataset", "series", "sha256", "start", "length", "codec"})
        if (!column.contains(name)) throw std::runtime_error("segment request column missing: " + name);
    struct Measured {
        std::string sha, codec;
        size_t start, length;
        uint64_t raw;
    };
    std::vector<Measured> measured;
    const std::string dataset = data_dir.filename().string();
    std::unordered_set<std::string> seen;
    size_t row_number = 1;
    const uint64_t inf = std::numeric_limits<uint64_t>::max();
    const auto parse_position = [&](const std::string &text, const char *name) {
        const std::string context = "segment request row " + std::to_string(row_number) + " " + name;
        if (text.empty()) throw std::runtime_error(context + " must be a nonnegative integer");
        size_t value = 0;
        for (const char ch : text) {
            if (ch < '0' || ch > '9')
                throw std::runtime_error(context + " must be a nonnegative integer: " + text);
            const size_t digit = static_cast<size_t>(ch - '0');
            if (value > (std::numeric_limits<size_t>::max() - digit) / 10)
                throw std::runtime_error(context + " is too large");
            value = value * 10 + digit;
        }
        return value;
    };
    while (std::getline(in, line)) {
        ++row_number;
        if (line.empty()) continue;
        if (line.back() == '\r') line.pop_back();
        const auto fields = AlpCostTable::split_csv(line);
        if (fields.size() != header.size())
            throw std::runtime_error("malformed segment request row " + std::to_string(row_number));
        if (fields[column.at("dataset")] != dataset || fields[column.at("series")] != series) continue;
        const auto &name = fields[column.at("codec")];
        const auto it = codec_index.find(name);
        if (it == codec_index.end()) throw std::runtime_error("unknown or excluded requested codec: " + name);
        const size_t start = parse_position(fields[column.at("start")], "start");
        const size_t length = parse_position(fields[column.at("length")], "length");
        if (length == 0 || start > x.size() || length > x.size() - start)
            throw std::runtime_error("segment request outside series at row " + std::to_string(row_number));
        const std::string key = std::to_string(start) + "," + std::to_string(length) + "," + name;
        if (!seen.insert(key).second)
            throw std::runtime_error("duplicate segment request: " + key);
        auto &codec = selected[it->second];
        const uint64_t raw = codec.oracle_cost_bits(x.data() + start, length, cost_model, rle_header_bits);
        if (raw == inf)
            throw std::runtime_error("missing exact requested cost: " + key + "; regenerate the requested ALP table");
        if (raw > inf - metadata_bits) throw std::overflow_error("requested segment cost overflow");
        if (!codec.verified_roundtrip(x.data() + start, length))
            throw std::runtime_error("requested segment bit-exact roundtrip failed: " + key);
        measured.push_back({fields[column.at("sha256")], name, start, length, raw});
    }
    if (in.bad()) throw std::runtime_error("failed reading segment requests: " + request_path.string());
    if (measured.empty()) throw std::runtime_error("no segment requests for " + dataset + "/" + series);
    if (output_path.has_parent_path()) fs::create_directories(output_path.parent_path());
    std::ofstream out(output_path);
    if (!out) throw std::runtime_error("cannot write requested segment costs: " + output_path.string());
    out << "dataset,series,sha256,start,length,codec,payload_bits,total_bits,cost_model,metadata_bits,rle_header_bits,roundtrip_ok,feasible\n";
    for (const auto &row : measured)
        out << dataset << ',' << series << ',' << row.sha << ',' << row.start << ','
            << row.length << ',' << row.codec << ',' << row.raw << ',' << row.raw + metadata_bits
            << ',' << oracle_cost_model_name(cost_model) << ',' << metadata_bits << ','
            << rle_header_bits << ",1,1\n";
    if (!out) throw std::runtime_error("failed writing requested segment costs");
    std::cerr << "requested segment costs: " << measured.size() << " verified rows; "
              << output_path.string() << '\n';
    return 0;
}

// Re-measure a proposed plan under the oracle's cold-start cost definition.
// This is deliberately separate from the oracle search: no measured costs are
// used to change the proposed boundaries or codecs.
static int score_plan(const fs::path &data_dir, const std::string &series,
                      size_t max_values, InputMode input_mode,
                      size_t decimal_digits, OracleCostModel cost_model,
                      size_t metadata_bits, uint64_t rle_header_bits,
                      const std::vector<std::string> &codec_order,
                      const fs::path &alp_cost_table_path,
                      const fs::path &plan_path, const std::string &plan_method,
                      const fs::path &results_path,
                      const std::string &tail_codec, size_t tail_max_values,
                      size_t plan_grid_len, AlpAlignment plan_alp_alignment) {
    if (series.empty() || plan_path.empty() || results_path.empty())
        throw std::invalid_argument("plan scoring requires --series, --plan and --plan-results");
    if (codec_order.empty())
        throw std::invalid_argument("plan scoring requires an explicit --codec-order");
    const auto path = resolve_series_file(data_dir, series);
    const auto x = read_series(path, max_values, input_mode, decimal_digits);
    if (x.empty()) throw std::runtime_error("empty plan-scoring series: " + path.string());

    const bool wants_alp = std::find(codec_order.begin(), codec_order.end(), "alp") != codec_order.end();
    std::vector<std::string> stream_names;
    for (const auto &name : codec_order) if (name != "alp") stream_names.push_back(name);
    std::vector<OracleCodec> selected;
    if (!stream_names.empty()) {
        for (auto &codec : codecs(stream_names)) selected.emplace_back(std::move(codec));
    }
    if (wants_alp) {
        if (alp_cost_table_path.empty())
            throw std::invalid_argument("plan codec pool includes alp but no --plan-alp-cost-table was given");
        selected.emplace_back(std::make_shared<const AlpCostTable>(AlpCostTable::load(alp_cost_table_path)));
    }
    std::map<std::string, size_t> codec_index;
    for (size_t i = 0; i < selected.size(); ++i) {
        selected[i].set_series_base(x.data());
        codec_index.emplace(selected[i].name(), i);
    }

    std::ifstream in(plan_path);
    if (!in) throw std::runtime_error("cannot open plan: " + plan_path.string());
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty plan: " + plan_path.string());
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = AlpCostTable::split_csv(line);
    const std::vector<std::string> required = {"dataset", "series", "method", "start", "length", "codec"};
    std::map<std::string, size_t> column;
    for (size_t i = 0; i < header.size(); ++i) column.emplace(header[i], i);
    for (const auto &name : required)
        if (!column.contains(name)) throw std::runtime_error("plan column missing: " + name);
    std::map<std::string, std::vector<PlanSegment>> plans;
    const std::string dataset = data_dir.filename().string();
    size_t row_number = 1;
    while (std::getline(in, line)) {
        ++row_number;
        if (line.empty()) continue;
        if (line.back() == '\r') line.pop_back();
        const auto fields = AlpCostTable::split_csv(line);
        if (fields.size() != header.size())
            throw std::runtime_error("malformed plan CSV row " + std::to_string(row_number));
        if (fields[column.at("dataset")] != dataset || fields[column.at("series")] != series)
            continue;
        const auto &method = fields[column.at("method")];
        if (!plan_method.empty() && method != plan_method) continue;
        plans[method].push_back({std::stoull(fields[column.at("start")]),
                                 std::stoull(fields[column.at("length")]),
                                 fields[column.at("codec")]});
    }
    if (plans.empty())
        throw std::runtime_error("plan has no rows for " + dataset + "/" + series +
                                 (plan_method.empty() ? "" : "/" + plan_method));
    if (results_path.has_parent_path()) fs::create_directories(results_path.parent_path());
    std::ofstream out(results_path);
    if (!out) throw std::runtime_error("cannot write plan scores: " + results_path.string());
    out << "dataset,series,method,n,segments,total_bits,bits_per_value,cost_model,tail_values,tail_codec\n";
    const uint64_t inf = std::numeric_limits<uint64_t>::max();
    for (auto &[method, segments] : plans) {
        std::sort(segments.begin(), segments.end(), [](const auto &a, const auto &b) {
            return a.start < b.start;
        });
        size_t next = 0;
        uint64_t total = 0;
        size_t segment_count = 0;
        for (const auto &segment : segments) {
            if (segment.length == 0 || segment.start != next ||
                segment.start > x.size() || segment.length > x.size() - segment.start)
                throw std::runtime_error("plan has a gap, overlap or invalid length at " +
                                         std::to_string(segment.start) + " for " + method);
            if (plan_grid_len && (segment.start % plan_grid_len != 0 ||
                                  segment.length % plan_grid_len != 0))
                throw std::runtime_error("plan segment is outside the requested grid at " +
                                         std::to_string(segment.start) + " for " + method);
            if (segment.codec == "alp" && plan_alp_alignment == AlpAlignment::Series &&
                segment.start % 1024 != 0)
                throw std::runtime_error("ALP plan start is not series-aligned at " +
                                         std::to_string(segment.start) + " for " + method);
            const auto it = codec_index.find(segment.codec);
            if (it == codec_index.end())
                throw std::runtime_error("unknown or excluded plan codec: " + segment.codec);
            const uint64_t raw = selected[it->second].oracle_cost_bits(
                x.data() + segment.start, segment.length, cost_model, rle_header_bits);
            if (raw == inf)
                throw std::runtime_error("no exact cost for plan segment " + method + " " +
                    std::to_string(segment.start) + "," + std::to_string(segment.length) +
                    "," + segment.codec + "; regenerate the ALP cost table if needed");
            if (raw > inf - metadata_bits || total > inf - raw - metadata_bits)
                throw std::overflow_error("plan cost overflow");
            total += raw + metadata_bits;
            next += segment.length;
            ++segment_count;
        }
        const size_t tail_values = x.size() - next;
        std::string chosen_tail_codec;
        if (tail_values != 0) {
            if (tail_codec.empty() || tail_values > tail_max_values)
                throw std::runtime_error("plan has an uncovered suffix beyond the allowed tail: " + method);
            uint64_t raw = inf;
            if (tail_codec == "best") {
                for (auto &codec : selected) {
                    if (codec.is_alp()) continue;
                    const uint64_t candidate = codec.oracle_cost_bits(
                        x.data() + next, tail_values, cost_model, rle_header_bits);
                    if (candidate < raw) {
                        raw = candidate;
                        chosen_tail_codec = codec.name();
                    }
                }
            } else {
                const auto it = codec_index.find(tail_codec);
                if (it == codec_index.end() || selected[it->second].is_alp())
                    throw std::runtime_error("plan tail codec must be a selected streaming codec");
                raw = selected[it->second].oracle_cost_bits(
                    x.data() + next, tail_values, cost_model, rle_header_bits);
                chosen_tail_codec = tail_codec;
            }
            if (raw == inf || raw > inf - metadata_bits || total > inf - raw - metadata_bits)
                throw std::overflow_error("plan tail cost overflow");
            total += raw + metadata_bits;
            ++segment_count;
        }
        out << dataset << ',' << series << ',' << method << ',' << x.size() << ','
            << segment_count << ',' << total << ',' << std::setprecision(12)
            << double(total) / double(x.size()) << ',' << oracle_cost_model_name(cost_model)
            << ',' << tail_values << ',' << chosen_tail_codec << '\n';
        std::cerr << "scored plan " << dataset << '/' << series << '/' << method
                  << ": " << total << " bits in " << segment_count << " segments\n";
    }
    if (!out) throw std::runtime_error("failed writing plan scores");
    return 0;
}

// A single continuous prefix pass supplies warm labels.  Restart calibration
// re-encodes only a short prefix at each candidate boundary, never all segment
// lengths.  The planner can therefore pay a cold-start correction once per
// chosen segment instead of silently charging it once per feature window.
static int write_warm_window_labels(const fs::path &data_dir, const std::string &series,
                                    size_t max_values, InputMode input_mode,
                                    size_t decimal_digits,
                                    const std::vector<std::string> &codec_order,
                                    size_t window, size_t restart_window,
                                    const fs::path &output_path) {
    if (series.empty() || output_path.empty() || window == 0 ||
        restart_window < window || restart_window % window != 0)
        throw std::invalid_argument("warm labels require --series, --warm-labels, and valid window lengths");
    const auto x = read_series(resolve_series_file(data_dir, series), max_values,
                               input_mode, decimal_digits);
    if (x.size() < window) throw std::runtime_error("series is shorter than warm window");
    std::vector<std::string> streaming;
    for (const auto &name : codec_order)
        if (name != "alp") streaming.push_back(name);
    if (streaming.empty()) throw std::invalid_argument("warm labels require streaming codecs");
    if (output_path.has_parent_path()) fs::create_directories(output_path.parent_path());
    std::ofstream out(output_path);
    if (!out) throw std::runtime_error("cannot write warm labels: " + output_path.string());
    out << "dataset,series,start,length,codec,warm_bits,restart_delta_bits,restart_length,cold_restart_bits,warm_restart_bits,restart_delta_by_window\n";
    const std::string dataset = data_dir.filename().string();
    const size_t covered = x.size() / window * window;
    for (auto &codec : codecs(streaming)) {
        std::vector<uint64_t> prefix;
        codec.prefix_bits(x.data(), x.size(), prefix);
        if (prefix.size() != x.size())
            throw std::runtime_error("incomplete warm prefix for " + codec.name());
        for (size_t start = 0; start < covered; start += window) {
            const uint64_t before = start ? prefix[start - 1] : 0;
            const size_t span = std::min(restart_window, covered - start);
            if (prefix[start + window - 1] < before ||
                prefix[start + span - 1] < before)
                throw std::runtime_error("non-monotone warm prefix for " + codec.name());
            const uint64_t warm = prefix[start + window - 1] - before;
            uint64_t cold_span = 0;
            uint64_t warm_span = 0;
            int64_t delta = 0;
            std::ostringstream deltas;
            for (size_t length = window; length <= span; length += window) {
                warm_span = prefix[start + length - 1] - before;
                cold_span = codec.oracle_cost_bits(
                    x.data() + start, length, OracleCostModel::Payload, 0);
                delta = static_cast<int64_t>(cold_span) -
                        static_cast<int64_t>(warm_span);
                if (length != window) deltas << '|';
                deltas << delta;
            }
            out << dataset << ',' << series << ',' << start << ',' << window << ','
                << codec.name() << ',' << warm << ',' << delta << ',' << span << ','
                << cold_span << ',' << warm_span << ',' << deltas.str() << '\n';
        }
    }
    if (!out) throw std::runtime_error("failed writing warm labels");
    std::cerr << "warm labels written: " << output_path.string() << '\n';
    return 0;
}

// The oracle already computes the exact cost of every candidate edge.  Keep
// those values in an append-only label table so a later run can add a new
// series, grid, codec, or cost model without rewriting the old labels.
class OracleEdgeTableAppender {
    static constexpr const char *kHeader =
        "dataset,input_variant,series,n,start,length,codec,cost_model,min_len,max_len,length_grid,"
        "metadata_bits,rle_header_bits,codec_bits,total_bits,bits_per_value,edge_status";

    fs::path path_;
    // Keeping every full CSV key as a std::string makes a large append-only
    // table consume several times its on-disk size (especially under ASan).
    // Two independent 64-bit digests retain practical duplicate protection
    // while keeping the resident set compact enough for dense grid sweeps.
    struct DigestHash {
        size_t operator()(const std::array<uint64_t, 2> &digest) const noexcept {
            return static_cast<size_t>(digest[0] ^ (digest[1] + 0x9e3779b97f4a7c15ULL
                                                     + (digest[0] << 6) + (digest[0] >> 2)));
        }
    };
    std::unordered_set<std::array<uint64_t, 2>, DigestHash> key_digests_;
    std::ofstream out_;
    size_t appended_ = 0;

    static std::vector<std::string> split_csv(const std::string &line) {
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) fields.push_back(field);
        return fields;
    }

    static std::array<uint64_t, 2> digest(const std::vector<std::string> &fields) {
        // The first 13 fields describe the candidate and its accounting
        // configuration. Costs are deliberately excluded: rerunning the
        // same key must not duplicate a row merely because it is rechecked.
        constexpr uint64_t fnv_prime = 1099511628211ULL;
        uint64_t a = 1469598103934665603ULL;
        uint64_t b = 1099511628211ULL ^ 0xd6e8feb86659fd93ULL;
        for (size_t i = 0; i < 13 && i < fields.size(); ++i) {
            for (const unsigned char c : fields[i]) {
                a ^= c;
                a *= fnv_prime;
                b ^= static_cast<uint64_t>(c) + 0x9e3779b97f4a7c15ULL;
                b *= 0x100000001b3ULL;
                b ^= b >> 29;
            }
            a ^= '|';
            a *= fnv_prime;
            b ^= 0x7f;
            b *= 0x100000001b3ULL;
        }
        return {a, b};
    }

public:
    explicit OracleEdgeTableAppender(const fs::path &path) : path_(path) {
        if (path_.empty()) return;
        if (path_.has_parent_path()) fs::create_directories(path_.parent_path());
        const bool existing = fs::exists(path_) && fs::file_size(path_) != 0;
        if (existing) {
            std::ifstream in(path_);
            if (!in) throw std::runtime_error("cannot open oracle edge table: " + path_.string());
            std::string line;
            if (!std::getline(in, line) || line != kHeader)
                throw std::runtime_error("invalid oracle edge table header: " + path_.string());
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                const auto fields = split_csv(line);
                if (fields.size() != 17)
                    throw std::runtime_error("invalid oracle edge table row in " + path_.string());
                key_digests_.insert(digest(fields));
            }
        }
        out_.open(path_, std::ios::out | std::ios::app);
        if (!out_) throw std::runtime_error("cannot append oracle edge table: " + path_.string());
        if (!existing) out_ << kHeader << '\n';
    }

    bool enabled() const { return !path_.empty(); }
    size_t appended() const { return appended_; }

    void append(const std::string &dataset, const std::string &input_variant,
                const std::string &series, size_t n, size_t min_len, size_t max_len,
                bool every_multiple, OracleCostModel cost_model,
                size_t metadata_bits, uint64_t rle_header_bits,
                const std::vector<OracleCodec> &selected,
                const std::vector<size_t> &lengths,
                const std::vector<std::vector<std::vector<uint64_t>>> &edge) {
        if (!enabled()) return;
        const std::string grid = every_multiple ? "multiples" : "powers-of-two";
        const std::string model = oracle_cost_model_name(cost_model);
        const uint64_t inf = std::numeric_limits<uint64_t>::max();
        for (size_t ci = 0; ci < selected.size(); ++ci) {
            for (size_t li = 0; li < lengths.size(); ++li) {
                const size_t length = lengths[li];
                for (size_t start_unit = 0; start_unit < edge[ci][li].size(); ++start_unit) {
                    const uint64_t total = edge[ci][li][start_unit];
                    if (total == inf) continue;
                    if (total < metadata_bits)
                        throw std::logic_error("oracle edge is smaller than metadata charge");
                    const uint64_t codec_bits = total - metadata_bits;
                    const size_t start = start_unit * min_len;
                    const double bpv = length ? double(codec_bits) / double(length) : 0.0;
                    std::ostringstream line;
                    line << dataset << ',' << input_variant << ',' << series << ',' << n << ','
                         << start << ',' << length << ',' << selected[ci].name() << ',' << model << ','
                         << min_len << ',' << max_len << ',' << grid << ',' << metadata_bits << ','
                         << rle_header_bits << ',' << codec_bits << ',' << total << ','
                         << std::setprecision(12) << bpv << ",computed\n";
                    const auto fields = split_csv(line.str());
                    if (fields.size() != 17) throw std::logic_error("oracle edge CSV assembly failed");
                    if (!key_digests_.insert(digest(fields)).second) continue;
                    out_ << line.str();
                    ++appended_;
                }
            }
        }
        out_.flush();
        if (!out_) throw std::runtime_error("failed while appending oracle edge table: " + path_.string());
    }
};

static int run_oracle(const fs::path &data_dir, size_t max_values,
                      const std::string &codec_filter, const std::string &series_filter,
                      InputMode input_mode, size_t decimal_digits,
                      size_t min_len, size_t max_len,
                      bool every_multiple,
                      AlpAlignment alp_alignment,
                      OracleCostModel cost_model,
                      size_t metadata_bits_per_segment,
                      size_t fixed_block_len,
                      uint64_t rle_header_bits,
                      const fs::path &results_path, const fs::path &report_path,
                      const std::vector<std::string> &codec_order,
                      const fs::path &alp_cost_table_path,
                      const fs::path &oracle_edge_cost_path,
                      bool baselines_only = false) {
    const auto lengths = oracle_lengths(min_len, max_len, every_multiple);
    const auto files = discover_series_files(data_dir, series_filter);

    const std::string dataset = data_dir.filename().string();
    std::shared_ptr<const AlpCostTable> alp_table;
    if (!alp_cost_table_path.empty()) {
        if (!every_multiple || min_len == 0 || (1024 % min_len) != 0)
            throw std::invalid_argument("ALP oracle requires --oracle-length-grid multiples with a min length dividing 1024");
        if (alp_alignment == AlpAlignment::Series && (1024 % min_len) != 0)
            throw std::invalid_argument("series-aligned ALP requires a min length dividing 1024");
        alp_table = std::make_shared<const AlpCostTable>(AlpCostTable::load(alp_cost_table_path));
    }

    std::vector<std::string> baseline_order;
    bool wants_alp = false;
    if (codec_order.empty()) {
        wants_alp = !alp_cost_table_path.empty() && (codec_filter.empty() || codec_filter == "alp");
        if (codec_filter.empty() || codec_filter != "alp") baseline_order = {};
        if (!codec_filter.empty() && codec_filter != "alp") baseline_order = {codec_filter};
    } else {
        for (const auto &name : codec_order) {
            if (name == "alp") wants_alp = true;
            else baseline_order.push_back(name);
        }
        if (!codec_filter.empty() && codec_filter != "alp") {
            baseline_order.erase(std::remove_if(baseline_order.begin(), baseline_order.end(),
                                                [&](const std::string &name) { return name != codec_filter; }),
                                 baseline_order.end());
        }
        if (codec_filter == "alp") wants_alp = true;
    }
    if (wants_alp && !alp_table)
        throw std::invalid_argument("codec pool requests alp but --oracle-alp-cost-table was not supplied");

    std::vector<OracleCodec> selected;
    if (!baseline_order.empty() || (codec_order.empty() && codec_filter.empty())) {
        auto all_codecs = codecs(baseline_order);
        for (auto &codec : all_codecs) selected.emplace_back(std::move(codec));
    }
    if (wants_alp) selected.emplace_back(alp_table);
    if (selected.empty()) throw std::invalid_argument("oracle codec filter selected no codec");

    std::vector<OracleRow> rows;
    OracleEdgeTableAppender edge_table(baselines_only ? fs::path{} : oracle_edge_cost_path);
    std::ofstream report(report_path);
    const bool has_rle = std::any_of(selected.begin(), selected.end(), [](const auto &codec) {
        return codec.name().rfind("rle+", 0) == 0;
    });
    const bool has_alp = std::any_of(selected.begin(), selected.end(), [](const auto &codec) {
        return codec.is_alp();
    });
    report << "# " << ((has_rle || has_alp) ? "E2" : "E1")
           << " lossless segmentation headroom pilot\n\n";
    report << "- Dataset directory: `" << data_dir.string() << "`\n";
    report << "- Input variant: " << input_mode_name(input_mode) << "\n";
    report << "- Cost model: " << oracle_cost_model_name(cost_model) << "\n";
    report << "- Codec pool: ";
    for (size_t i = 0; i < selected.size(); ++i) {
        if (i) report << ", ";
        report << selected[i].name();
    }
    report << "\n";
    if (has_alp) {
        report << "- ALP cost table: `" << alp_cost_table_path.string()
               << "` (official ALP-PDE payload bits; candidates are 64-value-aligned starts and 1024-value-multiple lengths)\n";
        report << "- ALP start alignment: " << alp_alignment_name(alp_alignment);
        if (alp_alignment == AlpAlignment::Series)
            report << " (starts must be multiples of 1024 values from the series origin)";
        else
            report << " (starts may use the fine 64-value grid)";
        report << "\n";
        report << "- ALP accounting: complete vector segments use the table cost; no ALP padding is charged, and a final non-vector tail falls back to the best selected non-ALP codec.\n";
        report << "- The configurable fixed-block baseline uses ALP only when the block length is a multiple of 1024; non-vector-compatible blocks (such as 240 values) use the streaming/RLE pool. A final remainder uses the best non-ALP fallback.\n";
    }
    if (input_mode == InputMode::Decimal) report << "- Decimal digits: " << decimal_digits << "\n";
    if (every_multiple) {
        report << "- Candidate lengths: every multiple of " << min_len << " from "
               << min_len << " through " << max_len
               << "; a terminal aligned remainder is also allowed\n";
    } else {
        report << "- Candidate lengths: powers of two from " << min_len << " through "
               << max_len << "; a terminal aligned remainder is also allowed\n";
    }
    if (cost_model == OracleCostModel::Payload) {
        report << "- Edge cost: codec payload bits before close/finalization, with no close padding or RLE transport header, plus "
               << metadata_bits_per_segment << " metadata bits per segment; the validation harness separately checks each codec's prefix accounting.\n";
    } else if (cost_model == OracleCostModel::ByteAligned) {
        report << "- Edge cost: codec payload rounded up to whole bytes per segment, with no close/finalization charge or RLE transport header, plus "
               << metadata_bits_per_segment << " metadata bits per segment; the validation harness separately checks each codec's prefix accounting.\n";
    } else {
        report << "- Edge cost: exact closed `encoded_bits()` plus "
               << metadata_bits_per_segment << " metadata bits per segment; the validation harness separately checks each codec's prefix accounting.\n";
    }
    if (has_alp)
        report << "- ALP exception: ALP edges use the official vector payload from the cost table under every model; ALP has no legacy close marker or RLE header.\n";
    report << "- Metadata charging: " << metadata_bits_per_segment
           << " bits are charged to every fixed-block, whole-series, and DP segment.\n";
    if (has_rle && cost_model == OracleCostModel::Current) {
        report << "- RLE accounting: `encoded_bits()` includes Gamma run lengths and the inner stream, "
               << "but excludes the standalone 56-byte RLE transport header. "
               << rle_header_bits << " additional bits are charged to each `rle+` segment.\n";
    } else if (has_rle) {
        report << "- RLE accounting: the standalone RLE transport header and inner-codec close/finalization are not charged in this cost model.\n";
    }
    if (baselines_only)
        report << "- Baselines-only mode: no adaptive edge grid, labels, or DP search is constructed.\n\n";
    else
        report << "- DP objective: minimize the selected edge cost plus metadata over aligned candidate segments; a final aligned remainder may use any length up to the maximum candidate, and a shorter tail is encoded as one segment.\n\n";
    if (edge_table.enabled()) {
        report << "- Edge label table: `" << oracle_edge_cost_path.string()
               << "` (append-only; existing candidate keys are skipped)\n";
    }

    for (const auto &file : files) {
        const auto x = read_series(file, max_values, input_mode, decimal_digits);
        const size_t n = x.size();
        if (n == 0) continue;
        for (auto &codec : selected) codec.set_series_base(x.data());
        const size_t covered = n - (n % min_len);
        if (covered < min_len) throw std::runtime_error("series is shorter than oracle minimum length");
        const size_t units = covered / min_len;
        std::cerr << "oracle pilot " << file.filename().string() << " (n=" << n << ")\n";

        const uint64_t inf = std::numeric_limits<uint64_t>::max();
        const auto stream_cost = [&](OracleCodec &codec, const double *values,
                                     size_t count) {
            return codec.oracle_cost_bits(values, count, cost_model, rle_header_bits);
        };
        const auto best_fallback = [&](const double *values, size_t count,
                                       std::string &codec_name) -> uint64_t {
            uint64_t best = inf;
            codec_name.clear();
            for (auto &codec : selected) {
                if (codec.is_alp()) continue;
                const uint64_t raw = stream_cost(codec, values, count);
                if (raw < best) {
                    best = raw;
                    codec_name = codec.name();
                }
            }
            return best;
        };
        std::vector<std::vector<std::vector<uint64_t>>> edge;
        size_t prefix_samples = 0;
        if (!baselines_only) {
        edge.assign(selected.size(), std::vector<std::vector<uint64_t>>(
            lengths.size(), std::vector<uint64_t>(units + 1, inf)));
        for (size_t ci = 0; ci < selected.size(); ++ci) {
            auto &codec = selected[ci];
            if (every_multiple) {
                if (codec.is_alp()) {
                    for (size_t start_unit = 0; start_unit < units; ++start_unit) {
                        if (alp_alignment == AlpAlignment::Series &&
                            ((start_unit * min_len) % 1024 != 0))
                            continue;
                        for (size_t li = 0; li < lengths.size(); ++li) {
                            const size_t length = lengths[li];
                            const size_t length_units = length / min_len;
                            if (start_unit + length_units > units) continue;
                            const uint64_t raw_cost = stream_cost(
                                codec, x.data() + start_unit * min_len, length);
                            if (raw_cost == inf ||
                                raw_cost > inf - metadata_bits_per_segment) continue;
                            edge[ci][li][start_unit] = raw_cost + metadata_bits_per_segment;
                            if (start_unit == 0) ++prefix_samples;
                        }
                    }
                    continue;
                }
                // For the dense grid, build one streaming prefix per aligned
                // start. Each finalized stream differs from its prefix by a
                // codec-specific constant, so this avoids re-encoding every
                // (start, length) pair independently.
                for (size_t start_unit = 0; start_unit < units; ++start_unit) {
                    const size_t remaining_values = (units - start_unit) * min_len;
                    const size_t prefix_length = std::min(max_len, remaining_values);
                    std::vector<uint64_t> prefix;
                    codec.prefix_bits(x.data() + start_unit * min_len,
                                      prefix_length, prefix);
                    for (size_t li = 0; li < lengths.size(); ++li) {
                        const size_t length = lengths[li];
                        const size_t length_units = length / min_len;
                        if (start_unit + length_units > units) continue;
                        if (prefix.size() < length)
                            throw std::runtime_error("oracle prefix accounting failed for " + codec.name());
                        const uint64_t prefix_cost = prefix[length - 1];
                        uint64_t raw_cost = prefix_cost;
                        if (cost_model == OracleCostModel::ByteAligned) {
                            if (raw_cost > std::numeric_limits<uint64_t>::max() - 7)
                                throw std::overflow_error("oracle byte-alignment overflow");
                            raw_cost = ((raw_cost + 7) / 8) * 8;
                        } else if (cost_model == OracleCostModel::Current) {
                            const uint64_t finalization = codec.finalization_overhead_bits();
                            const uint64_t format = codec.format_overhead_bits(rle_header_bits);
                            if (finalization > std::numeric_limits<uint64_t>::max() - format)
                                throw std::overflow_error("oracle finalization overhead overflow");
                            const uint64_t overhead = finalization + format;
                            if (raw_cost > std::numeric_limits<uint64_t>::max() - overhead)
                                throw std::overflow_error("oracle edge cost overflow");
                            raw_cost += overhead;
                        }
                        if (raw_cost > std::numeric_limits<uint64_t>::max() - metadata_bits_per_segment)
                            throw std::overflow_error("oracle edge cost overflow");
                        edge[ci][li][start_unit] = raw_cost + metadata_bits_per_segment;
                        if (start_unit == 0) ++prefix_samples;
                    }
                }
            } else {
                for (size_t li = 0; li < lengths.size(); ++li) {
                    const size_t length = lengths[li];
                    const size_t length_units = length / min_len;
                    if (length > covered) continue;
                    for (size_t start_unit = 0; start_unit + length_units <= units; ++start_unit) {
                        const uint64_t raw_cost = stream_cost(codec,
                            x.data() + start_unit * min_len, length);
                        if (raw_cost > std::numeric_limits<uint64_t>::max() - metadata_bits_per_segment)
                            throw std::overflow_error("oracle edge cost overflow");
                        edge[ci][li][start_unit] = raw_cost + metadata_bits_per_segment;
                    }
                    if (edge[ci][li][0] != inf) {
                        std::vector<uint64_t> prefix;
                        codec.prefix_bits(x.data(), length, prefix);
                        if (prefix.empty() || prefix.back() > edge[ci][li][0])
                            throw std::runtime_error("oracle prefix accounting failed for " + codec.name());
                        ++prefix_samples;
                    }
                }
            }
        }

        edge_table.append(dataset, input_mode_name(input_mode), file.stem().string(), n,
                          min_len, max_len, every_multiple, cost_model,
                          metadata_bits_per_segment, rle_header_bits,
                          selected, lengths, edge);
        }

        uint64_t best_whole_bits = inf;
        std::string best_whole_codec;
        size_t best_whole_segments = 0;
        for (auto &codec : selected) {
            uint64_t total = inf;
            size_t segment_count = 1;
            std::string reported_codec = codec.name();
            if (codec.is_alp()) {
                // A series can exceed ALP's maximum 102400-value row group.
                // Treat consecutive row groups as one outer ALP stream: sum
                // their official payloads, but charge segment metadata only
                // once. A final sub-vector tail uses one non-ALP segment.
                total = metadata_bits_per_segment;
                size_t alp_length = 0;
                while (n - alp_length >= 1024) {
                    size_t rowgroup_length = 0;
                    uint64_t rowgroup_bits = 0;
                    if (!codec.largest_alp_prefix(x.data() + alp_length,
                                                  n - alp_length, cost_model,
                                                  rowgroup_length, rowgroup_bits))
                        break;
                    if (rowgroup_length == 0 || total > inf - rowgroup_bits)
                        throw std::overflow_error("ALP whole-series cost overflow");
                    total += rowgroup_bits;
                    alp_length += rowgroup_length;
                }
                if (alp_length == 0) continue;
                if (alp_length < n) {
                    std::string tail_codec;
                    const uint64_t tail_raw = best_fallback(x.data() + alp_length,
                                                            n - alp_length, tail_codec);
                    if (tail_raw == inf || tail_raw > inf - metadata_bits_per_segment ||
                        total > inf - (tail_raw + metadata_bits_per_segment))
                        continue;
                    total += tail_raw + metadata_bits_per_segment;
                    ++segment_count;
                    reported_codec = "alp+tail/" + tail_codec;
                }
            } else {
                const uint64_t raw_cost = stream_cost(codec, x.data(), n);
                if (raw_cost == inf || raw_cost > inf - metadata_bits_per_segment)
                    continue;
                total = raw_cost + metadata_bits_per_segment;
            }
            rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                            "whole_series", reported_codec, n, n, segment_count, n, total});
            if (total < best_whole_bits) {
                best_whole_bits = total;
                best_whole_codec = reported_codec;
                best_whole_segments = segment_count;
            }
        }

        uint64_t best_fixed_bits = inf;
        std::string best_fixed_codec;
        size_t best_fixed_len = 0;
        size_t best_fixed_segments = 0;
        for (size_t ci = 0; ci < selected.size(); ++ci) {
            auto &codec = selected[ci];
            if (codec.is_alp()) continue;
            for (size_t li = 0; li < lengths.size(); ++li) {
                const size_t length = lengths[li];
                const size_t length_units = length / min_len;
                if (length > covered) continue;
                uint64_t total = 0;
                size_t segment_count = 0;
                for (size_t start_unit = 0; start_unit < units; start_unit += length_units) {
                    const size_t actual_units = std::min(length_units, units - start_unit);
                    const uint64_t raw_cost = !baselines_only && actual_units == length_units
                        ? edge[ci][li][start_unit]
                        : stream_cost(codec, x.data() + start_unit * min_len,
                                      actual_units * min_len);
                    uint64_t cost = raw_cost;
                    if (baselines_only || actual_units != length_units) {
                        if (raw_cost > std::numeric_limits<uint64_t>::max() - metadata_bits_per_segment)
                            throw std::overflow_error("fixed-block edge cost overflow");
                        cost += metadata_bits_per_segment;
                    }
                    if (cost == inf)
                        throw std::logic_error("missing or overflowing fixed-block edge cost");
                    if (total > std::numeric_limits<uint64_t>::max() - cost)
                        throw std::overflow_error("fixed-block total overflow");
                    total += cost;
                    ++segment_count;
                    if (actual_units != length_units) break;
                }
                if (covered < n) {
                    const uint64_t raw_tail = stream_cost(codec, x.data() + covered, n - covered);
                    if (raw_tail > std::numeric_limits<uint64_t>::max() - metadata_bits_per_segment)
                        throw std::overflow_error("fixed tail cost overflow");
                    const uint64_t tail_cost = raw_tail + metadata_bits_per_segment;
                    if (total > std::numeric_limits<uint64_t>::max() - tail_cost)
                        throw std::overflow_error("fixed-block total overflow");
                    total += tail_cost;
                    ++segment_count;
                }
                rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                                "fixed", codec.name(), n, length, segment_count, covered, total});
                if (total < best_fixed_bits) {
                    best_fixed_bits = total;
                    best_fixed_codec = codec.name();
                    best_fixed_len = length;
                    best_fixed_segments = segment_count;
                }
            }
        }

        // ALP fixed-block baselines use complete ALP vectors only.  A final
        // remainder is charged to the best non-ALP fallback codec; it is not
        // padded into an artificial ALP vector.
        for (auto &codec : selected) {
            if (!codec.is_alp()) continue;
            for (const size_t length : lengths) {
                if (length % 1024 != 0) continue;
                const size_t full_segments = n / length;
                if (full_segments == 0) continue;
                uint64_t total = 0;
                bool valid = true;
                for (size_t segment = 0; segment < full_segments; ++segment) {
                    const uint64_t raw = stream_cost(codec, x.data() + segment * length, length);
                    if (raw == inf || raw > inf - metadata_bits_per_segment ||
                        total > inf - (raw + metadata_bits_per_segment)) {
                        valid = false;
                        break;
                    }
                    total += raw + metadata_bits_per_segment;
                }
                if (!valid) continue;
                const size_t covered_alp = full_segments * length;
                std::string reported_codec = codec.name();
                const size_t tail_length = n - covered_alp;
                if (tail_length != 0) {
                    std::string tail_codec;
                    const uint64_t tail_raw = best_fallback(x.data() + covered_alp,
                                                            tail_length, tail_codec);
                    if (tail_raw == inf || tail_raw > inf - metadata_bits_per_segment ||
                        total > inf - (tail_raw + metadata_bits_per_segment))
                        continue;
                    total += tail_raw + metadata_bits_per_segment;
                    reported_codec = "alp+tail/" + tail_codec;
                }
                const size_t segment_count = full_segments + (tail_length ? 1 : 0);
                rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                                "fixed_alp", reported_codec, n, length, segment_count,
                                covered_alp, total});
                if (total < best_fixed_bits) {
                    best_fixed_bits = total;
                    best_fixed_codec = reported_codec;
                    best_fixed_len = length;
                    best_fixed_segments = segment_count;
                }
            }
        }

        // The paper's 1-minute SRRL setting uses four-hour (240-value)
        // fixed blocks. Keep this baseline independent of the oracle grid.
        // Other sampling rates may pass a time-equivalent block length.
        if (fixed_block_len == 0)
            throw std::invalid_argument("oracle fixed block length must be positive");
        const std::string fixed_strategy = "fixed" + std::to_string(fixed_block_len);
        const std::string best_fixed_strategy = "best_fixed" + std::to_string(fixed_block_len);
        uint64_t best_fixed240_bits = inf;
        std::string best_fixed240_codec;
        size_t best_fixed240_segments = 0;
        std::map<std::string, size_t> best_fixed240_mix;
        for (auto &codec : selected) {
            if (codec.is_alp()) continue;
            uint64_t total = 0;
            size_t segment_count = 0;
            for (size_t start = 0; start < n; start += fixed_block_len) {
                const size_t length = std::min(fixed_block_len, n - start);
                const uint64_t raw_cost = stream_cost(codec, x.data() + start, length);
                if (raw_cost > std::numeric_limits<uint64_t>::max() - metadata_bits_per_segment)
                    throw std::overflow_error("fixed-block baseline cost overflow");
                const uint64_t cost = raw_cost + metadata_bits_per_segment;
                if (total > std::numeric_limits<uint64_t>::max() - cost)
                    throw std::overflow_error("fixed-block baseline total overflow");
                total += cost;
                ++segment_count;
            }
            rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                            fixed_strategy, codec.name(), n, fixed_block_len,
                            segment_count, n, total});
        }

        // The fixed-block baseline chooses the best codec independently for
        // each block. ALP participates only when the configured block is a
        // complete 1024-value multiple, because the upstream ALP format is a
        // vector codec. A final partial block always uses a non-ALP fallback.
        // Keep the per-codec rows above for diagnostics, but use this mixed
        // per-block cost for the actual baseline comparison.
        const bool fixed_block_alp_compatible =
            has_alp && fixed_block_len >= 1024 && fixed_block_len % 1024 == 0;
        best_fixed240_bits = 0;
        best_fixed240_segments = 0;
        best_fixed240_mix.clear();
        for (size_t start = 0; start < n; start += fixed_block_len) {
            const size_t length = std::min(fixed_block_len, n - start);
            uint64_t block_best = inf;
            std::string block_codec;
            for (auto &codec : selected) {
                if (codec.is_alp()) continue;
                const uint64_t raw_cost = stream_cost(codec, x.data() + start, length);
                if (raw_cost > std::numeric_limits<uint64_t>::max() - metadata_bits_per_segment)
                    throw std::overflow_error("fixed-block cost overflow");
                const uint64_t cost = raw_cost + metadata_bits_per_segment;
                if (cost < block_best) {
                    block_best = cost;
                    block_codec = codec.name();
                }
            }
            if (fixed_block_alp_compatible && length == fixed_block_len) {
                for (auto &codec : selected) {
                    if (!codec.is_alp()) continue;
                    const uint64_t raw_cost = stream_cost(
                        codec, x.data() + start, length);
                    if (raw_cost == inf ||
                        raw_cost > std::numeric_limits<uint64_t>::max() - metadata_bits_per_segment)
                        continue;
                    const uint64_t cost = raw_cost + metadata_bits_per_segment;
                    if (cost < block_best) {
                        block_best = cost;
                        block_codec = codec.name();
                    }
                }
            }
            if (block_best == inf) {
                best_fixed240_bits = inf;
                best_fixed240_segments = 0;
                best_fixed240_mix.clear();
                break;
            }
            if (best_fixed240_bits > inf - block_best)
                throw std::overflow_error("fixed-block mixed baseline total overflow");
            best_fixed240_bits += block_best;
            ++best_fixed240_segments;
            ++best_fixed240_mix[block_codec];
        }
        if (best_fixed240_segments != 0) {
            best_fixed240_codec = best_fixed240_mix.size() == 1
                ? best_fixed240_mix.begin()->first
                : "mixed";
            rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                            best_fixed_strategy, best_fixed240_codec, n, fixed_block_len,
                            best_fixed240_segments, n, best_fixed240_bits});
        }

        const uint64_t best_baseline_bits = std::min(
            best_fixed_bits, std::min(best_whole_bits, best_fixed240_bits));
        const std::string best_baseline_codec =
            best_whole_bits <= best_fixed_bits && best_whole_bits <= best_fixed240_bits
                ? best_whole_codec
                : (best_fixed240_bits < best_fixed_bits ? best_fixed240_codec : best_fixed_codec);
        const size_t best_baseline_block_len =
            best_whole_bits <= best_fixed_bits && best_whole_bits <= best_fixed240_bits
                ? n
                : (best_fixed240_bits < best_fixed_bits ? fixed_block_len : best_fixed_len);
        const size_t best_baseline_segments =
            best_whole_bits <= best_fixed_bits && best_whole_bits <= best_fixed240_bits
                ? best_whole_segments
                : (best_fixed240_bits < best_fixed_bits ? best_fixed240_segments : best_fixed_segments);
        if (baselines_only) {
            rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                            "best_fixed", best_fixed_codec, n, best_fixed_len,
                            best_fixed_segments, covered, best_fixed_bits});
            rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                            "best_baseline", best_baseline_codec, n,
                            best_baseline_block_len, best_baseline_segments, n, best_baseline_bits});
            report << "## " << file.stem().string() << " (n=" << n << ")\n\n"
                   << "- Best fixed baseline: " << best_fixed_codec << " / " << best_fixed_len
                   << " values (" << best_fixed_segments << " segments), " << best_fixed_bits << " bits\n"
                   << "- Best whole-series baseline: " << best_whole_codec << " / "
                   << best_whole_segments << " segments, " << best_whole_bits << " bits\n"
                   << "- Best " << fixed_block_len << "-value fixed baseline (best codec per block): "
                   << best_fixed240_codec << " / " << best_fixed240_segments << " segments, "
                   << best_fixed240_bits << " bits\n"
                   << "- Best available baseline: " << best_baseline_codec << " / "
                   << best_baseline_segments << " segments, " << best_baseline_bits << " bits\n\n";
            std::cerr << "baselines complete " << file.stem().string() << ": "
                      << best_baseline_bits << " bits\n";
            continue;
        }

        std::vector<uint64_t> dp(units + 1, inf);
        struct Back { size_t previous = 0, codec = 0, length = 0; };
        std::vector<Back> back(units + 1);
        dp[0] = 0;
        for (size_t start_unit = 0; start_unit < units; ++start_unit) {
            if (dp[start_unit] == inf) continue;
            for (size_t li = 0; li < lengths.size(); ++li) {
                const size_t length_units = lengths[li] / min_len;
                if (start_unit + length_units > units) continue;
                for (size_t ci = 0; ci < selected.size(); ++ci) {
                    const uint64_t cost = edge[ci][li][start_unit];
                    if (cost == inf || dp[start_unit] > inf - cost) continue;
                    const uint64_t candidate = dp[start_unit] + cost;
                    const size_t next = start_unit + length_units;
                    if (candidate < dp[next]) {
                        dp[next] = candidate;
                        back[next] = {start_unit, ci, li};
                    }
                }
            }
            // A fixed block layout may end with a partial block (for example,
            // 768 values after several 1024-value blocks).  Permit one exact
            // aligned terminal segment for that remainder; otherwise the DP
            // would be compared against a stronger baseline than it can
            // represent and could report negative headroom.
            const size_t remaining_units = units - start_unit;
            const size_t max_units = max_len / min_len;
            if (remaining_units > 0 && remaining_units <= max_units) {
                const size_t terminal_length = remaining_units * min_len;
                for (size_t ci = 0; ci < selected.size(); ++ci) {
                    if (selected[ci].is_alp()) continue;
                    const uint64_t raw_cost = stream_cost(selected[ci],
                        x.data() + start_unit * min_len, terminal_length);
                    if (raw_cost > std::numeric_limits<uint64_t>::max() - metadata_bits_per_segment)
                        continue;
                    const uint64_t cost = raw_cost + metadata_bits_per_segment;
                    if (cost == inf || dp[start_unit] > inf - cost) continue;
                    const uint64_t candidate = dp[start_unit] + cost;
                    if (candidate < dp[units]) {
                        dp[units] = candidate;
                        back[units] = {start_unit, ci, 0};
                    }
                }
            }
        }
        if (dp[units] == inf) throw std::runtime_error("oracle DP found no tiling");

        uint64_t oracle_bits = dp[units];
        size_t oracle_segments = 0;
        std::map<std::string, size_t> mix;
        std::map<size_t, size_t> length_mix;
        struct SegmentChoice { size_t start = 0, length = 0; std::string codec; };
        std::vector<SegmentChoice> oracle_path;
        for (size_t at = units; at > 0;) {
            const Back b = back[at];
            ++oracle_segments;
            ++mix[selected[b.codec].name()];
            ++length_mix[(at - b.previous) * min_len];
            oracle_path.push_back({b.previous * min_len,
                                   (at - b.previous) * min_len,
                                   selected[b.codec].name()});
            at = b.previous;
        }
        // Backtracking collects the prefix segments from right to left.  Put
        // them in time order before appending the terminal remainder, so the
        // report lists segments by increasing start offset.
        std::reverse(oracle_path.begin(), oracle_path.end());
        if (covered < n) {
            uint64_t tail = inf;
            std::string tail_codec;
            for (auto &codec : selected) {
                if (codec.is_alp()) continue;
                const uint64_t raw_cost = stream_cost(codec, x.data() + covered, n - covered);
                if (raw_cost > std::numeric_limits<uint64_t>::max() - metadata_bits_per_segment)
                    continue;
                const uint64_t cost = raw_cost + metadata_bits_per_segment;
                if (cost < tail) { tail = cost; tail_codec = codec.name(); }
            }
            if (tail == inf || oracle_bits > inf - tail)
                throw std::overflow_error("oracle terminal cost overflow");
            oracle_bits += tail;
            ++oracle_segments;
            ++mix[tail_codec];
            ++length_mix[n - covered];
            oracle_path.push_back({covered, n - covered, tail_codec});
        }
        const uint64_t segmented_bits = oracle_bits;
        bool oracle_is_whole = false;
        if (best_whole_bits < oracle_bits) {
            oracle_bits = best_whole_bits;
            oracle_segments = best_whole_segments;
            mix.clear();
            mix[best_whole_codec] = 1;
            length_mix.clear();
            length_mix[n] = 1;
            oracle_path.clear();
            oracle_path.push_back({0, n, best_whole_codec});
            oracle_is_whole = true;
        }
        rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                        "oracle", oracle_is_whole ? best_whole_codec : "mixed", n,
                        oracle_is_whole ? n : 0, oracle_segments,
                        oracle_is_whole ? n : covered, oracle_bits});
        rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                        "best_fixed", best_fixed_codec, n, best_fixed_len,
                        best_fixed_segments, covered, best_fixed_bits});
        rows.push_back({dataset, input_mode_name(input_mode), file.stem().string(),
                        "best_baseline", best_baseline_codec, n,
                        best_baseline_block_len, best_baseline_segments,
                        n, best_baseline_bits});

        const double gain = best_baseline_bits
            ? 100.0 * (double(best_baseline_bits) - double(oracle_bits)) / double(best_baseline_bits)
            : 0.0;
        report << "## " << file.stem().string() << " (n=" << n << ")\n\n";
        report << "- Prefix samples checked: " << prefix_samples << "\n";
        report << "- Best fixed baseline: " << best_fixed_codec << " / " << best_fixed_len
               << " values (" << best_fixed_segments << " segments), " << best_fixed_bits << " bits\n";
        report << "- Best whole-series baseline: " << best_whole_codec << " / "
               << best_whole_segments << " segment" << (best_whole_segments == 1 ? "" : "s")
               << ", " << best_whole_bits << " bits\n";
        report << "- Best " << fixed_block_len
               << "-value fixed baseline (best codec per block): "
               << best_fixed240_codec << " / " << best_fixed240_segments
               << " segments, " << best_fixed240_bits << " bits\n";
        report << "- Best " << fixed_block_len << "-value codec mix: ";
        bool first = true;
        for (const auto &[name, count] : best_fixed240_mix) {
            if (!first) report << ", ";
            report << name << "=" << count;
            first = false;
        }
        report << "\n";
        report << "- Best available baseline: " << best_baseline_codec << " / "
               << best_baseline_segments << " segments, " << best_baseline_bits << " bits\n";
        report << "- Segmented DP cost before baseline comparison: " << segmented_bits << " bits\n";
        report << "- DP oracle: " << oracle_bits << " bits in " << oracle_segments
               << " segments\n";
        report << "- Oracle headroom over best available baseline: " << std::fixed << std::setprecision(3)
               << gain << "%\n";
        report << "- Oracle codec mix: ";
        first = true;
        for (const auto &[name, count] : mix) {
            if (!first) report << ", ";
            report << name << "=" << count;
            first = false;
        }
        report << "\n- Oracle length mix: ";
        first = true;
        for (const auto &[length, count] : length_mix) {
            if (!first) report << ", ";
            report << length << "=" << count;
            first = false;
        }
        report << "\n- Oracle segments (start,length,codec): ";
        first = true;
        for (const auto &segment : oracle_path) {
            if (!first) report << "; ";
            report << segment.start << "," << segment.length << "," << segment.codec;
            first = false;
        }
        report << "\n\n";
    }
    if (edge_table.enabled()) {
        report << "\n- Edge labels appended this run: " << edge_table.appended() << "\n";
    }
    write_oracle_csv(results_path, rows);
    return 0;
}

} // namespace harness

int main(int argc, char **argv) {
    try {
        fs::path data = "../Data";
        size_t max_values = 10000;
        size_t segments = 1000;
        uint64_t seed = 2323;
        std::string codec_filter;
        std::vector<std::string> codec_order;
        std::string series_filter;
        bool unsafe_gorilla = false;
        fs::path results_path;
        fs::path report_path;
        fs::path oracle_results_path;
        fs::path oracle_report_path;
        fs::path oracle_alp_cost_table_path;
        fs::path oracle_edge_cost_path;
        fs::path plan_path;
        fs::path plan_results_path;
        fs::path plan_alp_cost_table_path;
        fs::path segment_requests_path;
        fs::path segment_costs_path;
        fs::path warm_labels_path;
        std::string plan_method;
        std::string plan_tail_codec;
        size_t plan_tail_max_values = 0;
        size_t plan_grid_len = 0;
        size_t warm_window = 64;
        size_t restart_window = 256;
        harness::AlpAlignment plan_alp_alignment = harness::AlpAlignment::Series;
        bool run_oracle_mode = false;
        bool oracle_baselines_only = false;
        size_t oracle_min_len = 64;
        size_t oracle_max_len = 1024;
        bool oracle_every_multiple = false;
        harness::AlpAlignment oracle_alp_alignment = harness::AlpAlignment::Grid;
        harness::OracleCostModel oracle_cost_model = harness::OracleCostModel::Payload;
        size_t oracle_metadata_bits = 0;
        size_t oracle_fixed_block_len = 240;
        uint64_t oracle_rle_header_bits = 448;
        bool run_edge_tests = true;
        harness::InputMode input_mode = harness::InputMode::IntAsDouble;
        size_t decimal_digits = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&](const char *name) -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
                return argv[++i];
            };
            if (arg == "--data-dir") data = value("--data-dir");
            else if (arg == "--max-values") max_values = std::stoull(value("--max-values"));
            else if (arg == "--segments") segments = std::stoull(value("--segments"));
            else if (arg == "--seed") seed = std::stoull(value("--seed"));
            else if (arg == "--codec") codec_filter = value("--codec");
            else if (arg == "--codec-order") {
                std::string order = value("--codec-order");
                std::stringstream ss(order);
                std::string name;
                while (std::getline(ss, name, ',')) {
                    if (!name.empty()) codec_order.push_back(name);
                }
                if (codec_order.empty()) throw std::invalid_argument("empty --codec-order");
            }
            else if (arg == "--series") series_filter = value("--series");
            else if (arg == "--unsafe-gorilla") unsafe_gorilla = true;
            else if (arg == "--results") results_path = value("--results");
            else if (arg == "--report") report_path = value("--report");
            else if (arg == "--oracle") run_oracle_mode = true;
            else if (arg == "--oracle-baselines-only") {
                run_oracle_mode = true;
                oracle_baselines_only = true;
            }
            else if (arg == "--oracle-min-len") oracle_min_len = std::stoull(value("--oracle-min-len"));
            else if (arg == "--oracle-max-len") oracle_max_len = std::stoull(value("--oracle-max-len"));
            else if (arg == "--oracle-length-grid") {
                const auto grid = value("--oracle-length-grid");
                if (grid == "powers-of-two") oracle_every_multiple = false;
                else if (grid == "multiples") oracle_every_multiple = true;
                else throw std::invalid_argument("unknown oracle length grid: " + grid);
            }
            else if (arg == "--oracle-alp-alignment") {
                const auto alignment = value("--oracle-alp-alignment");
                if (alignment == "grid") oracle_alp_alignment = harness::AlpAlignment::Grid;
                else if (alignment == "series") oracle_alp_alignment = harness::AlpAlignment::Series;
                else throw std::invalid_argument("unknown ALP alignment: " + alignment);
            }
            else if (arg == "--oracle-overhead-model") {
                const auto model = value("--oracle-overhead-model");
                if (model == "payload") oracle_cost_model = harness::OracleCostModel::Payload;
                else if (model == "byte-aligned") oracle_cost_model = harness::OracleCostModel::ByteAligned;
                else if (model == "current") oracle_cost_model = harness::OracleCostModel::Current;
                else throw std::invalid_argument("unknown oracle overhead model: " + model);
            }
            else if (arg == "--oracle-metadata-bits") oracle_metadata_bits = std::stoull(value("--oracle-metadata-bits"));
            else if (arg == "--oracle-fixed-block-len") oracle_fixed_block_len = std::stoull(value("--oracle-fixed-block-len"));
            else if (arg == "--oracle-rle-header-bits") oracle_rle_header_bits = std::stoull(value("--oracle-rle-header-bits"));
            else if (arg == "--oracle-results") oracle_results_path = value("--oracle-results");
            else if (arg == "--oracle-report") oracle_report_path = value("--oracle-report");
            else if (arg == "--oracle-alp-cost-table") oracle_alp_cost_table_path = value("--oracle-alp-cost-table");
            else if (arg == "--oracle-edge-cost-table") oracle_edge_cost_path = value("--oracle-edge-cost-table");
            else if (arg == "--plan") plan_path = value("--plan");
            else if (arg == "--plan-method") plan_method = value("--plan-method");
            else if (arg == "--plan-results") plan_results_path = value("--plan-results");
            else if (arg == "--plan-alp-cost-table") plan_alp_cost_table_path = value("--plan-alp-cost-table");
            else if (arg == "--segment-requests") segment_requests_path = value("--segment-requests");
            else if (arg == "--segment-costs") segment_costs_path = value("--segment-costs");
            else if (arg == "--plan-tail-codec") plan_tail_codec = value("--plan-tail-codec");
            else if (arg == "--plan-tail-max-values") plan_tail_max_values = std::stoull(value("--plan-tail-max-values"));
            else if (arg == "--plan-grid-len") plan_grid_len = std::stoull(value("--plan-grid-len"));
            else if (arg == "--plan-alp-alignment") {
                const auto alignment = value("--plan-alp-alignment");
                if (alignment == "grid") plan_alp_alignment = harness::AlpAlignment::Grid;
                else if (alignment == "series") plan_alp_alignment = harness::AlpAlignment::Series;
                else throw std::invalid_argument("unknown plan ALP alignment: " + alignment);
            }
            else if (arg == "--warm-labels") warm_labels_path = value("--warm-labels");
            else if (arg == "--warm-window") warm_window = std::stoull(value("--warm-window"));
            else if (arg == "--restart-window") restart_window = std::stoull(value("--restart-window"));
            else if (arg == "--no-edge-test") run_edge_tests = false;
            else if (arg == "--input-mode") {
                const auto mode = value("--input-mode");
                if (mode == "raw-bits") input_mode = harness::InputMode::RawBits;
                else if (mode == "int-as-double") input_mode = harness::InputMode::IntAsDouble;
                else if (mode == "decimal") input_mode = harness::InputMode::Decimal;
                else throw std::invalid_argument("unknown input mode: " + mode);
            }
            else if (arg == "--decimal-digits") decimal_digits = std::stoull(value("--decimal-digits"));
            else if (arg == "--help") { std::cout << "codec_harness [--data-dir DIR] [--max-values N] [--segments N] [--seed N] [--codec NAME] [--codec-order A,B,C,D,alp] [--series NAME] [--input-mode raw-bits|int-as-double|decimal] [--decimal-digits D] [--unsafe-gorilla] [--no-edge-test] [--results FILE] [--report FILE] [--warm-labels FILE --warm-window N --restart-window N] [--oracle|--oracle-baselines-only --oracle-min-len N --oracle-max-len N --oracle-length-grid powers-of-two|multiples --oracle-alp-alignment grid|series --oracle-overhead-model payload|byte-aligned|current --oracle-metadata-bits N --oracle-fixed-block-len N --oracle-rle-header-bits N --oracle-alp-cost-table FILE --oracle-edge-cost-table FILE --oracle-results FILE --oracle-report FILE] [--plan FILE --plan-method NAME --plan-results FILE --plan-alp-cost-table FILE --plan-alp-alignment grid|series --plan-tail-codec CODEC|best --plan-tail-max-values N --plan-grid-len N] [--segment-requests FILE --segment-costs FILE --plan-alp-cost-table FILE]\n"; return 0; }
            else throw std::invalid_argument("unknown argument: " + arg);
        }
        if (results_path.empty()) results_path = data.parent_path() / "harness" / "results.csv";
        if (report_path.empty()) report_path = data.parent_path() / "harness" / "VALIDATION.md";
        if (!segment_requests_path.empty() || !segment_costs_path.empty()) {
            if (run_oracle_mode || !plan_path.empty() || !warm_labels_path.empty())
                throw std::invalid_argument("requested segment costs cannot be combined with --oracle, --plan or --warm-labels");
            return harness::write_requested_segment_costs(data, series_filter, max_values,
                input_mode, decimal_digits, oracle_cost_model, oracle_metadata_bits,
                oracle_rle_header_bits, codec_order, plan_alp_cost_table_path,
                segment_requests_path, segment_costs_path);
        }
        if (!warm_labels_path.empty()) {
            if (run_oracle_mode || !plan_path.empty())
                throw std::invalid_argument("--warm-labels cannot be combined with --plan or --oracle");
            return harness::write_warm_window_labels(data, series_filter, max_values,
                input_mode, decimal_digits, codec_order, warm_window,
                restart_window, warm_labels_path);
        }
        if (!plan_path.empty()) {
            if (run_oracle_mode) throw std::invalid_argument("--plan and --oracle cannot be combined");
            return harness::score_plan(data, series_filter, max_values, input_mode,
                                       decimal_digits, oracle_cost_model,
                                       oracle_metadata_bits, oracle_rle_header_bits,
                                       codec_order, plan_alp_cost_table_path,
                                       plan_path, plan_method, plan_results_path,
                                       plan_tail_codec, plan_tail_max_values,
                                       plan_grid_len, plan_alp_alignment);
        }
        if (run_oracle_mode) {
            if (oracle_results_path.empty()) oracle_results_path = data.parent_path() / "harness" / "oracle.csv";
            if (oracle_report_path.empty()) oracle_report_path = data.parent_path() / "harness" / "ORACLE.md";
            return harness::run_oracle(data, max_values, codec_filter, series_filter,
                                       input_mode, decimal_digits, oracle_min_len,
                                       oracle_max_len, oracle_every_multiple,
                                       oracle_alp_alignment,
                                       oracle_cost_model,
                                       oracle_metadata_bits, oracle_fixed_block_len,
                                       oracle_rle_header_bits,
                                       oracle_results_path,
                                       oracle_report_path, codec_order,
                                       oracle_alp_cost_table_path,
                                       oracle_edge_cost_path, oracle_baselines_only);
        }
        return harness::run(data, max_values, segments, seed, codec_filter, series_filter,
                            unsafe_gorilla, results_path, report_path, run_edge_tests,
                            input_mode, decimal_digits, codec_order);
    } catch (const std::exception &e) {
        std::cerr << "codec_harness: " << e.what() << '\n';
        return 2;
    }
}
