// Replay only the supplied, frozen full-series plan. No search, learning,
// implicit tail, or old cost-table lookup is performed by this executable.
#ifdef _WIN32
using uint = unsigned int; // legacy TSXor uses the POSIX uint alias
#endif
#define main original_codec_harness_main_not_called
#include "../codec_harness.cpp"
#undef main
#include "alp_payload.hpp"
#include <cctype>
#include <chrono>
#include <cstring>
#include <set>

namespace frozen {
using framed::Bytes;
constexpr std::uint32_t mixed_method = 3;
constexpr std::uint64_t file_header_bytes = 24, frame_header_bytes = 12;
const std::map<std::string, std::uint32_t> codec_ids{
    {"gorilla",16},{"chimp",17},{"chimp128",18},{"tsxor",19},
    {"rle+tsxor",23},{"alp",24}};
// The omitted IDs 20..22 stay reserved; retained payload IDs are NOT renumbered.
std::uint32_t parse_action(const std::string& name) {
    const auto codec=codec_ids.find(name);
    if(codec==codec_ids.end()) throw std::runtime_error("codec is outside the strict-eight public pool: "+name);
    return codec->second;
}
std::string codec_name(std::uint32_t id) {
    for (const auto& [name, candidate] : codec_ids) if (id == candidate) return name;
    throw std::runtime_error("persisted codec id is outside the strict-eight public pool");
}
std::string quote(const std::string& text) {
    std::string result = "\"";
    for (unsigned char ch : text) {
        if (ch == '\\' || ch == '"') { result += '\\'; result += ch; }
        else if (ch == '\n') result += "\\n";
        else if (ch == '\r') result += "\\r";
        else if (ch == '\t') result += "\\t";
        else if (ch < 32) result += '?';
        else result += ch;
    }
    return result + '"';
}
std::uint64_t number(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("plan start/length must be nonnegative decimal integers");
    std::size_t used = 0;
    auto result = std::stoull(text, &used);
    if (used != text.size()) throw std::runtime_error("invalid plan integer");
    return result;
}
Bytes read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open input: " + path.string());
    const auto length = in.tellg();
    if (length < 0 || static_cast<std::uint64_t>(length) > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("invalid input size");
    Bytes result(static_cast<std::size_t>(length));
    in.seekg(0);
    if (!result.empty() && !in.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size())))
        throw std::runtime_error("cannot completely read input");
    return result;
}
void protect_outputs(const std::vector<fs::path>& outputs, const std::vector<fs::path>& inputs) {
    std::set<fs::path> unique;
    for (const auto& path : outputs) {
        const auto canonical = fs::weakly_canonical(path);
        if (!unique.insert(canonical).second || fs::exists(canonical))
            throw std::runtime_error("output paths must be distinct and nonexistent; existing files are preserved");
        for (const auto& input : inputs)
            if (canonical == fs::weakly_canonical(input)) throw std::runtime_error("output aliases an input");
    }
}
void write_file(const fs::path& path, const Bytes& bytes) {
    // C++23 noreplace is exclusive creation, including the file-exists race.
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::noreplace);
    if (!out) throw std::runtime_error("cannot exclusively create output: " + path.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) throw std::runtime_error("output write/close failed");
}

// Small independent SHA-256 implementation for binding the plan's optional
// checksum to the actual raw input. No external crypto library is required.
std::string sha256(const Bytes& data) {
    static constexpr std::uint32_t constants[64]{
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t,8> state{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    Bytes padded = data;
    if (data.size() > std::numeric_limits<std::uint64_t>::max()/8) throw std::runtime_error("SHA input size overflow");
    padded.push_back(128);
    while (padded.size() % 64 != 56) padded.push_back(0);
    framed::put(padded, static_cast<std::uint64_t>(data.size())*8, 8);
    for (std::size_t offset=0; offset<padded.size(); offset+=64) {
        std::array<std::uint32_t,64> words{};
        for (unsigned i=0;i<16;++i) for (unsigned j=0;j<4;++j) words[i]=(words[i]<<8)|padded[offset+i*4+j];
        for (unsigned i=16;i<64;++i) {
            const auto a=words[i-15], b=words[i-2];
            const auto s0=std::rotr(a,7)^std::rotr(a,18)^(a>>3), s1=std::rotr(b,17)^std::rotr(b,19)^(b>>10);
            words[i]=words[i-16]+s0+words[i-7]+s1;
        }
        auto [a,b,c,d,e,f,g,h]=state;
        for (unsigned i=0;i<64;++i) {
            const auto s1=std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25), ch=(e&f)^(~e&g);
            const auto t1=h+s1+ch+constants[i]+words[i];
            const auto s0=std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22), maj=(a&b)^(a&c)^(b&c), t2=s0+maj;
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
    }
    std::ostringstream out; out<<std::hex<<std::setfill('0');
    for (auto value:state) out<<std::setw(8)<<value;
    return out.str();
}
std::vector<std::string> csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted=false, closed=false;
    for (std::size_t i=0;i<line.size();++i) {
        char ch=line[i];
        if (quoted) {
            if (ch=='"') {
                if (i+1<line.size() && line[i+1]=='"') {field+='"';++i;}
                else {quoted=false;closed=true;}
            } else field+=ch;
        } else if (ch==',') {fields.push_back(field);field.clear();closed=false;}
        else if (ch=='\r' && i+1==line.size()) {}
        else if (closed) {if(ch!=' ' && ch!='\t') throw std::runtime_error("invalid quoted CSV field");}
        else if(ch=='"') {if(!field.empty()) throw std::runtime_error("invalid CSV quote");quoted=true;}
        else field+=ch;
    }
    if(quoted) throw std::runtime_error("multiline/unterminated quoted CSV fields are not supported");
    fields.push_back(field);return fields;
}
struct Segment {std::size_t start,length;std::uint32_t codec;};
std::vector<Segment> read_plan(const fs::path& path,const std::string& method,std::size_t values,const std::string& checksum) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot open frozen plan");
    std::string line;
    if(!std::getline(in,line)) throw std::runtime_error("empty frozen plan");
    if(line.compare(0,3,"\xEF\xBB\xBF")==0) line.erase(0,3);
    const auto header=csv(line);
    std::map<std::string,std::size_t> columns;
    for(std::size_t i=0;i<header.size();++i) if(!columns.emplace(header[i],i).second) throw std::runtime_error("duplicate frozen-plan CSV column");
    for(const auto& key:{"method","start","length","codec"}) if(!columns.contains(key)) throw std::runtime_error(std::string("frozen plan missing column ")+key);
    std::vector<Segment> result;std::string identity;std::size_t next=0;
    while(std::getline(in,line)) {
        if(line.empty()||line=="\r") continue;
        const auto fields=csv(line);
        if(fields.size()!=header.size()) throw std::runtime_error("frozen-plan CSV column count differs");
        if(fields[columns.at("method")]!=method) continue;
        std::string current;
        for(const auto& key:{"dataset","series"}) if(columns.contains(key)) current+=fields[columns.at(key)]+"\x1f";
        if(result.empty()) identity=current;
        else if(current!=identity) throw std::runtime_error("requested method spans multiple dataset/series identities; supply a one-series plan");
        if(columns.contains("sha256")) {
            auto digest=fields[columns.at("sha256")];
            std::transform(digest.begin(),digest.end(),digest.begin(),[](unsigned char ch){return static_cast<char>(std::tolower(ch));});
            if(!digest.empty() && digest!=checksum) throw std::runtime_error("frozen-plan SHA256 differs from raw input");
        }
        const auto start=number(fields[columns.at("start")]), length=number(fields[columns.at("length")]);
        if(start!=next || !length || start>values || length>values-start || length>100000000)
            throw std::runtime_error("frozen plan must cover raw input in order, without gaps, overlaps, empty segments, or implicit tails");
        const auto codec=parse_action(fields[columns.at("codec")]);
        if(codec==24) framed::validate_alp_count(length);
        result.push_back({static_cast<std::size_t>(start),static_cast<std::size_t>(length),codec});
        next+=length;
    }
    if(in.bad() || result.empty() || next!=values) throw std::runtime_error("frozen plan does not cover full input; no implicit tail fallback is allowed");
    return result;
}
std::vector<double> raw_values(const Bytes& raw) {
    if(raw.empty() || raw.size()%8) throw std::runtime_error("input must be nonempty headerless little-endian binary64");
    static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
    std::vector<double> result(raw.size()/8);
    for(std::size_t i=0;i<result.size();++i) {
        std::uint64_t value=0;for(unsigned j=0;j<8;++j)value|=std::uint64_t(raw[i*8+j])<<(j*8);
        result[i]=std::bit_cast<double>(value);
    }
    const auto first=std::bit_cast<std::uint64_t>(result[0]);
    if(first && first==result.size()-1) throw std::runtime_error("legacy count-prefixed input is not headerless float64");
    return result;
}
Bytes native_encode(std::uint32_t id,const double* values,std::size_t count) {
    const auto name=codec_name(id); // Pool guard precedes every native invocation.
    if(id==24) return framed::encode_alp(values,count);
    auto codecs=harness::codecs({name});
    const auto packet=codecs.front().encode(values,count);
    const auto kind=static_cast<harness::Kind>(id-15);
    const auto packed=harness::unpack(packet,kind,count);
    Bytes result;framed::put(result,packed.stream_bits,8);
    result.insert(result.end(),packed.bytes.begin(),packed.bytes.end());
    return result;
}
std::vector<double> native_decode(std::uint32_t id,const Bytes& bytes,std::size_t count) {
    const auto name=codec_name(id);
    if(id==24) return framed::decode_alp(bytes,count);
    if(bytes.size()<8) throw std::runtime_error("truncated streaming logical-length header");
    const auto stream_bits=harness::get_u64(bytes,0);
    Bytes stored(bytes.begin()+8,bytes.end());
    if(stored.empty() || stream_bits>static_cast<std::uint64_t>(stored.size())*8)
        throw std::runtime_error("invalid streaming logical bit count");
    harness::Packed packed{stream_bits,0,0,std::move(stored)};
    auto codecs=harness::codecs({name});
    return codecs.front().decode(harness::pack(static_cast<harness::Kind>(id-15),count,packed),count);
}
struct Decoded {
    Bytes raw;
    std::uint64_t values=0,blocks=0,codec_storage=0;
    std::uint64_t stream_transport=0,alp_blocks=0,alp_rd_blocks=0;
};
Decoded decode_file(const Bytes& file) {
    framed::Reader reader(file);
    for(char ch:std::string("F64BLK01")) if(reader.get(1)!=static_cast<unsigned char>(ch)) throw std::runtime_error("invalid complete-file magic");
    if(reader.get(4)!=mixed_method || reader.get(4)!=0) throw std::runtime_error("invalid mixed complete-file method/block size");
    Decoded result;result.values=reader.get(8);
    if(!result.values || result.values>std::numeric_limits<std::size_t>::max()/8) throw std::runtime_error("invalid complete-file value count");
    // Every decoded frame must consume actual file bytes before allocation.
    // Do not allocate from an untrusted total-value count in the file header.
    std::uint64_t decoded_values=0;
    while(decoded_values<result.values) {
        const auto id=static_cast<std::uint32_t>(reader.get(4));codec_name(id);
        const auto count=reader.get(4), bytes=reader.get(4);
        if(!count || count>result.values-decoded_values || !bytes || bytes>file.size()) throw std::runtime_error("invalid complete-file frame geometry");
        // Match the exporter's documented per-frame safety limit. In
        // particular, corrupt metadata must not request multi-gigabyte arrays.
        if(count>100000000) throw std::runtime_error("complete-file frame exceeds decoder safety limit");
        Bytes payload;payload.reserve(static_cast<std::size_t>(bytes));
        for(std::uint64_t i=0;i<bytes;++i) payload.push_back(static_cast<std::uint8_t>(reader.get(1)));
        auto values=native_decode(id,payload,static_cast<std::size_t>(count));
        if(values.size()!=count) throw std::runtime_error("codec returned a different decoded value count");
        for(double value:values) {
            const auto bits=std::bit_cast<std::uint64_t>(value);
            for(unsigned j=0;j<8;++j) result.raw.push_back(static_cast<std::uint8_t>(bits>>(j*8)));
        }
        result.codec_storage+=bytes;++result.blocks;decoded_values+=count;
        if(id<24) result.stream_transport+=8;
        else {
            ++result.alp_blocks;
            if(payload.size()>6 && payload[6]==static_cast<std::uint8_t>(alp::Scheme::ALP_RD)) ++result.alp_rd_blocks;
        }
    }
    reader.finish();return result;
}
std::string report(const Decoded& decoded,std::uint64_t file_bytes,std::uint64_t mismatches,bool verified,const std::string& operation) {
    const auto framing=file_header_bytes+frame_header_bytes*decoded.blocks;
    if(file_bytes!=framing+decoded.codec_storage) throw std::runtime_error("complete-file accounting identity failed");
    std::ostringstream out;
    out<<"{\n  \"status\": "<<quote(mismatches?"not_bitwise_lossless":"passed")
       <<",\n  \"operation\": "<<quote(operation)
       <<",\n  \"container_version\": \"F64BLK01\",\n  \"method_id\": 3,\n  \"values\": "<<decoded.values
       <<",\n  \"decoded_values\": "<<decoded.raw.size()/8<<",\n  \"mismatch_count\": "<<mismatches
       <<",\n  \"blocks\": "<<decoded.blocks<<",\n  \"complete_file_bytes\": "<<file_bytes
       <<",\n  \"complete_file_bits\": "<<file_bytes*8<<",\n  \"codec_storage_bytes\": "<<decoded.codec_storage
       <<",\n  \"framing_bytes\": "<<framing<<",\n  \"file_header_bytes\": 24,\n  \"frame_header_bytes\": 12"
       <<",\n  \"complete_file_roundtrip_verified\": "<<(verified?"true":"false")
       <<",\n  \"decoded_from_persisted_file_only\": true,\n  \"stream_adapter_metadata_bytes\": "<<decoded.stream_transport
       <<",\n  \"alp_blocks\": "<<decoded.alp_blocks<<",\n  \"alp_rd_blocks\": "<<decoded.alp_rd_blocks
       <<",\n  \"stream_payload_model\": \"stream_length8_plus_existing_codec_bytes\""
       <<",\n  \"alp_payload_model\": \"ALPF01_full_scheme_dictionary_counts_ffor_exceptions\""
       <<",\n  \"plan_changed\": false,\n  \"search_used\": false,\n  \"cost_table_used\": false\n}\n";
    return out.str();
}
void write_report(const fs::path& path,const std::string& text) {write_file(path,Bytes(text.begin(),text.end()));}
void self_test() {
    std::vector<double> values(1024);
    for(std::size_t i=0;i<values.size();++i) values[i]=1.+double((i/5)%113)/8.;
    unsigned checks=0, excluded_names=0, excluded_ids=0;
    for(const auto& [name,id]:codec_ids) {
        if(parse_action(name)!=id) throw std::runtime_error("self-test action ID changed");
        const auto payload=native_encode(id,values.data(),values.size());
        const auto decoded=native_decode(id,payload,values.size());
        if(decoded.size()!=values.size() || std::memcmp(values.data(),decoded.data(),values.size()*8))
            throw std::runtime_error("self-test retained codec failed bitwise roundtrip");
        ++checks;
    }
    for(const auto* name:{"rle+gorilla","rle+chimp","rle+chimp128"}) {
        bool rejected=false;
        try { (void)parse_action(name); } catch(const std::runtime_error&) { rejected=true; }
        if(!rejected) throw std::runtime_error("self-test accepted an excluded action");
        ++excluded_names;
    }
    for(const auto id:{20u,21u,22u}) {
        bool rejected=false;
        try { (void)codec_name(id); } catch(const std::runtime_error&) { rejected=true; }
        if(!rejected) throw std::runtime_error("self-test accepted an excluded codec ID");
        ++excluded_ids;
    }
    std::cout<<"{\"status\":\"passed\",\"mode\":\"self-test\",\"checks\":"<<checks
             <<",\"excluded_action_rejections\":"<<excluded_names
             <<",\"excluded_id_rejections\":"<<excluded_ids
             <<",\"retained_physical_ids\":[16,17,18,19,23,24],\"strict_eight_pool\":true}\n";
}
} // namespace frozen

int main(int argc,char** argv) {
    try {
        using namespace frozen;
        if(argc==2 && std::string(argv[1])=="--self-test") { self_test(); return 0; }
        if(argc==2 && std::string(argv[1])=="--help") {
            std::cout<<"FrozenPlanFile INPUT.raw PLAN.csv METHOD OUTPUT.f64blk REPORT.json\n"
                       "FrozenPlanFile --decode INPUT.f64blk OUTPUT.raw REPORT.json\n"
                       "FrozenPlanFile --self-test\n";return 0;
        }
        if(argc==5 && std::string(argv[1])=="--decode") {
            protect_outputs({argv[3],argv[4]},{argv[2]});
            const auto file=read_file(argv[2]);const auto decoded=decode_file(file);
            write_file(argv[3],decoded.raw);
            // A decode-only operation has no external original for comparison.
            // It is a verified self-contained decode, not a fresh lossless claim.
            const auto text=report(decoded,file.size(),0,false,"standalone_decode");
            write_report(argv[4],text);std::cout<<text;return 0;
        }
        if(argc!=6) throw std::runtime_error("usage: FrozenPlanFile INPUT.raw PLAN.csv METHOD OUTPUT.f64blk REPORT.json; or --decode INPUT.f64blk OUTPUT.raw REPORT.json");
        protect_outputs({argv[4],argv[5]},{argv[1],argv[2]});
        const auto raw=read_file(argv[1]);const auto values=raw_values(raw);
        const auto plan=read_plan(argv[2],argv[3],values.size(),sha256(raw));
        Bytes file{'F','6','4','B','L','K','0','1'};
        framed::put(file,mixed_method,4);framed::put(file,0,4);framed::put(file,values.size(),8);
        for(const auto& segment:plan) {
            const auto payload=native_encode(segment.codec,values.data()+segment.start,segment.length);
            if(payload.empty() || payload.size()>std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("encoded frame length exceeds uint32");
            framed::put(file,segment.codec,4);framed::put(file,segment.length,4);framed::put(file,payload.size(),4);
            file.insert(file.end(),payload.begin(),payload.end());
        }
        write_file(argv[4],file);
        // Intentionally reload an independently owned byte buffer from disk.
        // No encoder object, raw value, plan, or cost table is passed to decode.
        const auto persisted=read_file(argv[4]);const auto decoded=decode_file(persisted);
        if(decoded.raw.size()!=raw.size()) throw std::runtime_error("complete-file decoded length differs from raw input");
        std::uint64_t mismatches=0;
        for(std::size_t offset=0;offset<raw.size();offset+=8)
            if(std::memcmp(raw.data()+offset,decoded.raw.data()+offset,8)) ++mismatches;
        const auto text=report(decoded,persisted.size(),mismatches,!mismatches,"frozen_plan_replay");
        write_report(argv[5],text);std::cout<<text;return mismatches?2:0;
    } catch(const std::exception& error) {
        std::cerr<<"FrozenPlanFile: "<<error.what()<<'\n';
        std::cout<<"{\"status\":\"failed\",\"complete_file_roundtrip_verified\":false,\"error\":"<<frozen::quote(error.what())<<"}\n";
        return 1;
    }
}
