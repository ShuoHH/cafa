// Two public fixed compositions built from native BtrBlocks primitives. This is NOT
// the complete BtrBlocks selector and is never linked to its original build.
// Input/output raw values are little-endian IEEE binary64. BTRFIX01 integers
// are big-endian; native codec payloads retain their upstream representation.
#include "compression/SchemePicker.hpp"
#ifndef BTR_FIXED_POLICY_V1
#error "BtrBlocksFixed must use the checked private fixed-policy source overlay"
#endif
#include "scheme/SchemeConfig.hpp"
#include "scheme/double/Pseudodecimal.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
#include <vector>

namespace {
using namespace btrblocks;
constexpr char kMagic[] = "BTRFIX01";
constexpr std::uint32_t kVersion = 1, kPhysicalBlock = 65536;
constexpr std::uint64_t kHeaderBytes = 24, kBlockHeaderBytes = 8;
const std::array<std::string, 5> kActions{{"btr_onevalue", "btr_dict_bp",
    "btr_rle_fixed", "btr_frequency_fixed", "btr_decimal_fixed"}};
const std::array<DoubleSchemeType, 5> kSchemes{{DoubleSchemeType::ONE_VALUE,
    DoubleSchemeType::DICT, DoubleSchemeType::RLE, DoubleSchemeType::FREQUENCY,
    DoubleSchemeType::PSEUDODECIMAL}};
// Preserve the five-slot wire numbering: only indices 1/4 (IDs 33/36) are public.
// Reserved slots must never invoke an excluded native double scheme.
constexpr bool retained_action(unsigned action) { return action == 1 || action == 4; }
void require_retained_action(unsigned action) {
  if (!retained_action(action))
    throw std::runtime_error("fixed action is outside the strict-eight public pool");
}
// Exactly the native Pseudodecimal lookup constants, not pow(10, -exponent).
// Adapted from BtrBlocks (MIT); see third_party_licenses/BtrBlocks-MIT.txt.
constexpr double kFractions[] = {1., .1, .01, .001, .0001, .00001, .000001,
    .0000001, .00000001, .000000001, .0000000001, .00000000001,
    .000000000001, .0000000000001, .00000000000001, .000000000000001,
    .0000000000000001, .00000000000000001, .000000000000000001,
    .0000000000000000001, .00000000000000000001,
    .000000000000000000001, .0000000000000000000001};

class Unsupported : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::string quote(const std::string& text) {
  std::string result = "\"";
  for (const unsigned char c : text) {
    if (c == '\\' || c == '"') { result += '\\'; result += c; }
    else if (c < 32) {
      constexpr char hex[] = "0123456789abcdef";
      result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
    } else result += c;
  }
  return result + '"';
}

// Refuse overwrite and symlinks. An incomplete newly created file is removed.
class ExclusiveFile {
 public:
  explicit ExclusiveFile(std::string path) : path_(std::move(path)) {
    descriptor_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (descriptor_ < 0) throw std::runtime_error("cannot exclusively create output: " + path_);
  }
  ExclusiveFile(const ExclusiveFile&) = delete;
  ExclusiveFile& operator=(const ExclusiveFile&) = delete;
  ~ExclusiveFile() {
    if (descriptor_ >= 0) ::close(descriptor_);
    if (!committed_) std::remove(path_.c_str());
  }
  void write(const void* data, std::size_t size) {
    const auto* cursor = static_cast<const unsigned char*>(data);
    while (size) {
      const auto count = ::write(descriptor_, cursor, size);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) throw std::runtime_error("failed writing output");
      cursor += count; size -= static_cast<std::size_t>(count);
    }
  }
  void finish() {
    if (::fsync(descriptor_) != 0) throw std::runtime_error("failed synchronizing output");
    const auto fd = descriptor_; descriptor_ = -1;
    if (::close(fd) != 0) throw std::runtime_error("failed closing output");
    committed_ = true;
  }
 private:
  std::string path_;
  int descriptor_ = -1;
  bool committed_ = false;
};

std::uint64_t read_integer(std::istream& input, unsigned width, bool little = false) {
  unsigned char bytes[8]{};
  if (!input.read(reinterpret_cast<char*>(bytes), width))
    throw std::runtime_error("truncated input/header");
  std::uint64_t result = 0;
  for (unsigned i = 0; i < width; ++i)
    result |= std::uint64_t(bytes[i]) << (8 * (little ? i : width - i - 1));
  return result;
}

void write_integer(ExclusiveFile& output, std::uint64_t value, unsigned width) {
  unsigned char bytes[8]{};
  for (unsigned i = 0; i < width; ++i)
    bytes[i] = static_cast<unsigned char>(value >> (8 * (width - i - 1)));
  output.write(bytes, width);
}

std::uint64_t bits(double value) {
  std::uint64_t result;
  std::memcpy(&result, &value, 8);
  return result;
}

void require_little_native() {
  const std::uint32_t one = 1;
  if (*reinterpret_cast<const unsigned char*>(&one) != 1)
    throw std::runtime_error("native BtrBlocks payload format requires a little-endian host");
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                "IEEE binary64 is required");
}

unsigned action_index(const std::string& action) {
  for (unsigned i = 0; i < kActions.size(); ++i)
    if (action == kActions[i]) { require_retained_action(i); return i; }
  throw std::runtime_error("unknown fixed action: " + action);
}

DoubleScheme& scheme(unsigned action) {
  require_retained_action(action);
  if (action >= kSchemes.size()) throw std::runtime_error("unknown action id");
  auto& pool = DoubleSchemePicker::MyTypeWrapper::getSchemes();
  const auto found = pool.find(kSchemes[action]);
  if (found == pool.end() || !found->second)
    throw std::runtime_error("required native double scheme is not registered");
  return *found->second;
}

void configure() {
  require_little_native();
  BtrBlocksConfig::configure([](BtrBlocksConfig& cfg) { cfg.block_size = kPhysicalBlock; });
  auto& config = SchemeConfig::get();
  config.doubles.rle_force_values_scheme = DoubleSchemeType::UNCOMPRESSED;
  config.doubles.rle_force_counts_scheme = IntegerSchemeType::BP;
  config.doubles.pseudodecimal_significant_digit_bits_limits = 31;
}

bool decimal_convertible(double value) {
  if (value == 0. && std::signbit(value)) return false;
  for (const auto fraction : kFractions) {
    const auto rounded = std::round(value / fraction);
    if (!std::isfinite(rounded) || rounded < 0. || rounded > 2147483647.) continue;
    const auto number = static_cast<std::uint64_t>(rounded);
    if (bits(static_cast<double>(number) * fraction) == bits(value)) return true;
  }
  return false;
}

void validate_domain(const std::vector<double>& input, unsigned action) {
  if (input.empty() || input.size() > kPhysicalBlock)
    throw std::runtime_error("invalid physical block count");
  std::size_t exceptions = 0;
  for (const auto value : input) {
    if (!std::isfinite(value))
      throw Unsupported("native fixed schemes explicitly reject NaN/Inf; no replacement codec");
    if (action == 0 && bits(value) != bits(input[0]))
      throw Unsupported("ONE_VALUE requires one identical binary64 bit pattern per physical block");
    if (action >= 1 && action <= 3 && value == 0. && std::signbit(value))
      throw Unsupported("native numeric equality can merge signed zeros; this fixed action rejects -0");
    if (action == 4 && !decimal_convertible(value)) ++exceptions;
  }
  // Native Decimal otherwise returns a fake oversize estimate without finishing
  // its payload. That is an inapplicable action, NEVER a serializable encoding.
  if (action == 4 && exceptions > input.size() / 2)
    throw Unsupported("native Pseudodecimal would abort: more than half the values are patches");
}

std::uint32_t maximum_payload(std::uint32_t count) {
  return count * 64 + 65536;
}

std::vector<unsigned char> encode_block(const std::vector<double>& input, unsigned action) {
  require_retained_action(action);
  validate_domain(input, action);
  auto stats = DoubleStats::generateStats(input.data(), nullptr, input.size());
  const auto capacity = maximum_payload(input.size());
  std::vector<unsigned char> encoded(capacity + SIMD_EXTRA_BYTES, 0);
  // Direct concrete scheme invocation: no top-level picker or isUsable/sample.
  const auto size = scheme(action).compress(input.data(), nullptr, encoded.data(), stats, 3);
  if (size == 0 || size > capacity)
    throw std::runtime_error("native fixed scheme returned an invalid payload length");
  encoded.resize(size);
  return encoded;
}

std::vector<double> decode_block(const std::vector<unsigned char>& encoded,
                                 std::uint32_t count, unsigned action) {
  require_retained_action(action);
  if (!count || count > kPhysicalBlock || encoded.empty() || encoded.size() > maximum_payload(count))
    throw std::runtime_error("invalid fixed block count/payload length");
  // Native SIMD decoders may overread/overwrite a few lanes. Padding is scratch
  // RAM only and is not persisted, discarded from neither size nor bit checks.
  std::vector<unsigned char> padded(encoded.size() + SIMD_EXTRA_BYTES, 0);
  std::copy(encoded.begin(), encoded.end(), padded.begin());
  std::vector<double> result(count + SIMD_EXTRA_ELEMENTS(DOUBLE) + 16, 0.);
  scheme(action).decompress(result.data(), nullptr, padded.data(), count, 0);
  result.resize(count);
  return result;
}

void equal_bits(const std::vector<double>& expected, const std::vector<double>& actual) {
  if (expected.size() != actual.size() ||
      std::memcmp(expected.data(), actual.data(), expected.size() * 8) != 0)
    throw std::runtime_error("native fixed scheme failed bitwise roundtrip; output is not eligible");
}

struct Report {
  unsigned action = 0;
  std::uint64_t values = 0, blocks = 0, payload_bytes = 0, complete_bytes = 0;
};

Report encode(const std::string& name, const std::string& source, const std::string& destination) {
  Report report; report.action = action_index(name);
  std::ifstream input(source, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open raw input");
  const auto length = input.tellg();
  if (length <= 0 || static_cast<std::uint64_t>(length) % 8)
    throw std::runtime_error("input must be nonempty raw binary64");
  const auto total = static_cast<std::uint64_t>(length) / 8;
  input.seekg(0);
  ExclusiveFile output(destination);
  output.write(kMagic, 8);
  write_integer(output, kVersion, 4);
  write_integer(output, report.action + 32, 4);
  write_integer(output, total, 8);
  while (report.values < total) {
    const auto count = static_cast<std::uint32_t>(std::min<std::uint64_t>(kPhysicalBlock, total - report.values));
    std::vector<double> values(count);
    for (auto& value : values) {
      const auto word = read_integer(input, 8, true);
      std::memcpy(&value, &word, 8);
    }
    const auto payload = encode_block(values, report.action);
    equal_bits(values, decode_block(payload, count, report.action));
    write_integer(output, count, 4);
    write_integer(output, payload.size(), 4);
    output.write(payload.data(), payload.size());
    report.values += count; report.payload_bytes += payload.size(); ++report.blocks;
  }
  if (input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("raw input changed during compression");
  output.finish();
  report.complete_bytes = kHeaderBytes + report.blocks * kBlockHeaderBytes + report.payload_bytes;
  return report;
}

Report decode(const std::string& source, const std::string& destination) {
  std::ifstream input(source, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open fixed file");
  const auto length = input.tellg();
  if (length < static_cast<std::streamoff>(kHeaderBytes)) throw std::runtime_error("truncated fixed file");
  input.seekg(0);
  char magic[8]{};
  if (!input.read(magic, 8) || std::memcmp(magic, kMagic, 8) || read_integer(input, 4) != kVersion)
    throw std::runtime_error("invalid BTRFIX01 magic/version");
  const auto code = read_integer(input, 4);
  if (code < 32 || code > 36) throw std::runtime_error("unknown fixed action in file");
  Report report; report.action = code - 32;
  require_retained_action(report.action); // Reject before creating a decoded file.
  const auto total = read_integer(input, 8);
  if (!total) throw std::runtime_error("empty fixed file");
  ExclusiveFile output(destination);
  while (report.values < total) {
    const auto count = read_integer(input, 4), size = read_integer(input, 4);
    const auto expected_count = std::min<std::uint64_t>(kPhysicalBlock, total - report.values);
    if (count != expected_count || !size || size > maximum_payload(count) ||
        size > static_cast<std::uint64_t>(length - input.tellg()))
      throw std::runtime_error("invalid/truncated fixed frame");
    std::vector<unsigned char> payload(size);
    if (!input.read(reinterpret_cast<char*>(payload.data()), size)) throw std::runtime_error("truncated native payload");
    const auto values = decode_block(payload, count, report.action);
    // Native payloads and raw IEEE binary64 are both little-endian on this host.
    output.write(values.data(), values.size() * 8);
    report.values += count; report.payload_bytes += size; ++report.blocks;
  }
  if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing fixed-file bytes");
  report.complete_bytes = kHeaderBytes + report.blocks * kBlockHeaderBytes + report.payload_bytes;
  if (report.complete_bytes != static_cast<std::uint64_t>(length)) throw std::runtime_error("fixed file size mismatch");
  output.finish();
  return report;
}

void self_test() {
  unsigned checks = 0;
  for (const auto count : {1u, 3u, 4u, 31u, 128u, 1024u, 65536u}) {
    for (const unsigned action : {1u, 4u}) {
      std::vector<double> values(count);
      for (unsigned i = 0; i < count; ++i) {
        switch (action) {
          case 0: values[i] = -0.; break;
          case 1: values[i] = double(i % 7) / 4.; break;
          case 2: values[i] = double((i / 13) % 5); break;
          case 3: values[i] = i % 11 ? 3.25 : double(i) / 4.; break;
          default: values[i] = double(i % 113) / 8.; break;
        }
      }
      const auto payload = encode_block(values, action);
      equal_bits(values, decode_block(payload, count, action)); ++checks;
    }
  }
  unsigned excluded_rejections = 0;
  for (const unsigned action : {0u, 2u, 3u}) {
    bool rejected = false;
    try { (void)action_index(kActions[action]); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("self-test accepted an excluded fixed action");
    ++excluded_rejections;
  }
  std::cout << "{\"status\":\"passed\",\"mode\":\"self-test\",\"checks\":" << checks
            << ",\"excluded_action_rejections\":" << excluded_rejections
            << ",\"retained_physical_ids\":[33,36],\"strict_eight_pool\":true"
            << ",\"adaptive_selection\":false,\"expansion_fallback\":false}\n";
}
} // namespace

int main(int argc, char** argv) {
  std::string action = "unknown";
  try {
    configure();
    if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
    Report report;
    std::string mode;
    if (argc == 5 && std::string(argv[1]) == "--encode") {
      action = argv[2]; mode = "encode"; report = encode(action, argv[3], argv[4]);
    } else if (argc == 4 && std::string(argv[1]) == "--decode") {
      mode = "decode"; report = decode(argv[2], argv[3]); action = kActions[report.action];
    } else throw std::runtime_error("usage: BtrBlocksFixed --encode ACTION INPUT.raw OUTPUT.bfx; --decode INPUT.bfx OUTPUT.raw; --self-test");
    std::cout << "{\"status\":\"passed\",\"method\":" << quote(action)
              << ",\"mode\":" << quote(mode) << ",\"values\":" << report.values
              << ",\"decoded_values\":" << report.values << ",\"blocks\":" << report.blocks
              << ",\"payload_bytes\":" << report.payload_bytes
              << ",\"complete_file_bytes\":" << report.complete_bytes
              << ",\"complete_file_bits\":" << report.complete_bytes * 8
              << ",\"container_version\":\"BTRFIX01\",\"adaptive_selection\":false"
              << ",\"expansion_fallback\":false}\n";
    return 0;
  } catch (const Unsupported& error) {
    std::cout << "{\"status\":\"unsupported\",\"method\":" << quote(action)
              << ",\"error\":" << quote(error.what()) << "}\n"; return 3;
  } catch (const std::exception& error) {
    std::cout << "{\"status\":\"failed\",\"method\":" << quote(action)
              << ",\"error\":" << quote(error.what()) << "}\n"; return 2;
  }
}
