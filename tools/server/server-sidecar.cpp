#include "server-sidecar.h"

#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

namespace {

constexpr double ETA = 0.05;
constexpr double EPSILON = 1e-12;

struct sha256_context {
    std::array<uint32_t, 8> state = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<uint8_t, 64> buffer{};
    uint64_t bit_count = 0;
    size_t buffer_size = 0;
};

constexpr std::array<uint32_t, 64> SHA256_K = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

uint32_t rotr(uint32_t value, uint32_t amount) {
    return (value >> amount) | (value << (32 - amount));
}

void sha256_transform(sha256_context & ctx, const uint8_t * block) {
    std::array<uint32_t, 64> w{};
    for (size_t i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[4*i]) << 24) |
               (uint32_t(block[4*i + 1]) << 16) |
               (uint32_t(block[4*i + 2]) << 8) |
               uint32_t(block[4*i + 3]);
    }
    for (size_t i = 16; i < w.size(); ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = ctx.state[0];
    uint32_t b = ctx.state[1];
    uint32_t c = ctx.state[2];
    uint32_t d = ctx.state[3];
    uint32_t e = ctx.state[4];
    uint32_t f = ctx.state[5];
    uint32_t g = ctx.state[6];
    uint32_t h = ctx.state[7];

    for (size_t i = 0; i < w.size(); ++i) {
        const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + s1 + ch + SHA256_K[i] + w[i];
        const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx.state[0] += a;
    ctx.state[1] += b;
    ctx.state[2] += c;
    ctx.state[3] += d;
    ctx.state[4] += e;
    ctx.state[5] += f;
    ctx.state[6] += g;
    ctx.state[7] += h;
}

void sha256_update(sha256_context & ctx, const uint8_t * data, size_t size) {
    ctx.bit_count += uint64_t(size) * 8;
    while (size > 0) {
        const size_t take = std::min(size, ctx.buffer.size() - ctx.buffer_size);
        std::memcpy(ctx.buffer.data() + ctx.buffer_size, data, take);
        ctx.buffer_size += take;
        data += take;
        size -= take;
        if (ctx.buffer_size == ctx.buffer.size()) {
            sha256_transform(ctx, ctx.buffer.data());
            ctx.buffer_size = 0;
        }
    }
}

std::string sha256_finish(sha256_context & ctx) {
    const uint64_t bit_count = ctx.bit_count;
    const uint8_t one = 0x80;
    sha256_update(ctx, &one, 1);
    const uint8_t zero = 0;
    while (ctx.buffer_size != 56) {
        sha256_update(ctx, &zero, 1);
    }
    std::array<uint8_t, 8> length{};
    for (size_t i = 0; i < length.size(); ++i) {
        length[7 - i] = uint8_t(bit_count >> (8*i));
    }
    sha256_update(ctx, length.data(), length.size());

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint32_t value : ctx.state) {
        out << std::setw(8) << value;
    }
    return out.str();
}

std::string sha256_bytes(const std::vector<uint8_t> & bytes) {
    sha256_context ctx;
    sha256_update(ctx, bytes.data(), bytes.size());
    return sha256_finish(ctx);
}

std::string sha256_string(const std::string & value) {
    return sha256_bytes(std::vector<uint8_t>(value.begin(), value.end()));
}

std::string fnv1a64_bytes(const uint8_t * data, size_t size) {
    uint64_t value = 1469598103934665603ull;
    for (size_t index = 0; index < size; ++index) {
        value ^= uint64_t(data[index]);
        value *= 0x100000001B3ull;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::vector<uint8_t> read_bytes(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open sidecar file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot determine sidecar file size: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty() && !input.read(reinterpret_cast<char *>(bytes.data()), size)) {
        throw std::runtime_error("cannot read sidecar file: " + path.string());
    }
    return bytes;
}

void write_bytes(const fs::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create sidecar file: " + path.string());
    }
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
    }
    if (!output) {
        throw std::runtime_error("cannot write sidecar file: " + path.string());
    }
}

// Prototype sidecars use fixed-width little-endian records so inactive slots
// occupy the same serialized space as active slots.
void append_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        bytes.push_back(uint8_t(value >> (8 * i)));
    }
}

uint64_t read_u64(const std::vector<uint8_t> & bytes, size_t & offset) {
    if (offset + sizeof(uint64_t) > bytes.size()) {
        throw std::runtime_error("truncated prototype metadata");
    }
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= uint64_t(bytes[offset++]) << (8 * i);
    }
    return value;
}

void append_double(std::vector<uint8_t> & bytes, double value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    append_u64(bytes, bits);
}

double read_double(const std::vector<uint8_t> & bytes, size_t & offset) {
    const uint64_t bits = read_u64(bytes, offset);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void append_float(std::vector<uint8_t> & bytes, float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    for (size_t i = 0; i < sizeof(bits); ++i) {
        bytes.push_back(uint8_t(bits >> (8 * i)));
    }
}

float read_float(const std::vector<uint8_t> & bytes, size_t & offset) {
    if (offset + sizeof(uint32_t) > bytes.size()) {
        throw std::runtime_error("truncated prototype vector payload");
    }
    uint32_t bits = 0;
    for (size_t i = 0; i < sizeof(bits); ++i) {
        bits |= uint32_t(bytes[offset++]) << (8 * i);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

double vector_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) {
        sum += double(value) * double(value);
    }
    return std::sqrt(sum);
}

double dot_product(const std::vector<float> & left, const float * right, size_t size) {
    if (left.size() != size) {
        throw std::runtime_error("sidecar component width does not match query width");
    }
    double result = 0.0;
    for (size_t i = 0; i < size; ++i) {
        result += double(left[i]) * double(right[i]);
    }
    return result;
}

double cosine_similarity(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("prototype vector widths do not match");
    }
    const double left_norm = vector_norm(left);
    const double right_norm = vector_norm(right);
    if (left_norm <= EPSILON || right_norm <= EPSILON) {
        return 0.0;
    }
    double dot = 0.0;
    for (size_t i = 0; i < left.size(); ++i) {
        dot += double(left[i]) * double(right[i]);
    }
    return dot / (left_norm * right_norm);
}

double cosine_similarity(const std::vector<float> & left, const float * right, size_t size) {
    if (left.size() != size) {
        throw std::runtime_error("prototype query width does not match centroid width");
    }
    const double left_norm = vector_norm(left);
    double right_norm_squared = 0.0;
    for (size_t i = 0; i < size; ++i) {
        right_norm_squared += double(right[i]) * double(right[i]);
    }
    const double right_norm = std::sqrt(right_norm_squared);
    if (left_norm <= EPSILON || right_norm <= EPSILON) {
        return 0.0;
    }
    return dot_product(left, right, size) / (left_norm * right_norm);
}

void normalize_vector(std::vector<float> & values) {
    const double norm = vector_norm(values);
    if (norm <= EPSILON) {
        return;
    }
    for (float & value : values) {
        value = float(double(value) / norm);
    }
}

std::string vector_checksum(const std::vector<float> & values) {
    const auto * begin = reinterpret_cast<const uint8_t *>(values.data());
    return fnv1a64_bytes(begin, values.size() * sizeof(float));
}

std::string vector_hash(const std::vector<float> & values) {
    return vector_checksum(values);
}

bool valid_id(const std::string & id) {
    if (id.empty() || id.size() > 128) {
        return false;
    }
    return std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_' || c == '.';
    });
}

void require_json_file(const json & manifest, const char * key, const std::string & expected) {
    if (!manifest.contains(key) || !manifest.at(key).is_string() || manifest.at(key).get<std::string>() != expected) {
        throw std::runtime_error(std::string("invalid sidecar manifest field: ") + key);
    }
}

template <typename T>
T required_json(const json & value, const char * key) {
    if (!value.contains(key)) {
        throw std::runtime_error(std::string("missing sidecar field: ") + key);
    }
    try {
        return value.at(key).get<T>();
    } catch (const std::exception & error) {
        throw std::runtime_error(std::string("invalid sidecar field '") + key + "': " + error.what());
    }
}

bool close_enough(double left, double right) {
    return std::abs(left - right) <= 1e-9;
}

struct step30_counts {
    uint64_t total = 0;
    uint64_t matching = 0;
};

struct step30_gate {
    double novelty = 0.0;
    double surprise = 0.0;
    double recurrence = 0.0;
    double consistency = 0.0;
    double gate = 0.0;
};

step30_gate compute_step30_gate(const step30_counts & counts) {
    step30_gate result;
    result.novelty = counts.total == 0 ? 1.0 : 0.0;
    const double probability = counts.total == 0 ? 0.0 :
        static_cast<double>(counts.matching) / static_cast<double>(counts.total);
    result.surprise = 1.0 - probability;
    result.recurrence = std::min(1.0, static_cast<double>(counts.total) / 3.0);
    result.consistency = probability;
    result.gate = std::clamp(
        0.25 * std::max(result.novelty, result.surprise) +
        result.recurrence * result.consistency, 0.0, 1.0);
    return result;
}

} // namespace

struct server_sidecar_manager::sidecar {
    struct component {
        double coefficient = 0.0;
        std::string source;
        std::string destination;
        std::vector<float> value;
        std::vector<float> key;
    };

    struct prototype {
        bool active = false;
        uint64_t count = 0;
        double mean_coefficient = 0.0;
        double key_dispersion = 0.0;
        double q_coherence = 0.0;
        uint64_t creation_step = 0;
        uint64_t last_update_step = 0;
        uint64_t forced_merge_count = 0;
        std::vector<float> centroid_key;
        std::vector<float> mean_q;
    };

    struct prototype_update {
        size_t index = 0;
        double similarity = 0.0;
        bool created = false;
        bool forced_merge = false;
    };

    struct episode {
        std::vector<float> query;
        std::string source;
        llama_token baseline_greedy = -1;
        uint64_t version = 0;
    };

    mutable std::mutex mutex;
    std::string id;
    int32_t width = 0;
    bool learning_enabled = false;
    bool loaded = true;
    std::string memory_backend = "raw";
    int32_t configured_prototypes = 0;
    double merge_threshold = 0.90;
    int32_t top_cluster_candidates = 10;
    std::vector<component> components;
    // Prototype mode allocates every slot, including inactive slots, at
    // creation/load time. This is the fixed physical memory boundary.
    std::vector<prototype> prototypes;
    std::unordered_map<std::string, episode> episodes;
    std::unordered_map<std::string, step30_counts> writer_counts;
    uint64_t next_episode = 1;
    uint64_t version = 0;
    uint64_t retrievals = 0;
    uint64_t writes = 0;
    uint64_t lifetime_merges = 0;
    uint64_t forced_merges = 0;
    uint64_t total_created_prototypes = 0;
    uint64_t experiences_absorbed = 0;
    double sum_b = 0.0;
    double max_b = 0.0;
    std::string loaded_path;
    std::string package_sha;
    std::string base_model_identifier = "unavailable";
    std::string base_model_sha256;
    std::string injection_profile_id;
    uint64_t query_captures = 0;
    server_sidecar_dispatch_telemetry dispatch_telemetry;
    sha256_context state_hash_context;
    mutable bool prototype_hash_dirty = true;
    mutable std::string prototype_hash_cache;

    bool is_prototype_backend() const {
        return memory_backend == "prototype";
    }

    void initialize_prototype_slots(int32_t count) {
        if (count < 1) {
            throw std::runtime_error("prototype capacity must be positive");
        }
        configured_prototypes = count;
        prototypes.assign(size_t(count), prototype{});
        for (prototype & slot : prototypes) {
            slot.centroid_key.assign(size_t(width), 0.0f);
            slot.mean_q.assign(size_t(width), 0.0f);
        }
        prototype_hash_dirty = true;
        prototype_hash_cache.clear();
    }

    size_t active_prototype_count() const {
        size_t active = 0;
        for (const prototype & slot : prototypes) {
            active += slot.active ? 1 : 0;
        }
        return active;
    }

    uintmax_t serialized_file_bytes() const {
        if (loaded_path.empty()) {
            return 0;
        }
        const fs::path package(loaded_path);
        const std::array<const char *, 4> names = {
            "manifest.json", "prototypes.f32", "prototypes.meta", "state.meta",
        };
        uintmax_t total = 0;
        for (const char * name : names) {
            std::error_code error;
            const uintmax_t size = fs::file_size(package / name, error);
            if (!error) {
                total += size;
            }
        }
        return total;
    }

    std::vector<uint8_t> prototype_vectors_payload() const {
        std::vector<uint8_t> payload;
        payload.reserve(prototypes.size() * size_t(width) * 2 * sizeof(float));
        for (const prototype & slot : prototypes) {
            for (float value : slot.centroid_key) {
                append_float(payload, value);
            }
            for (float value : slot.mean_q) {
                append_float(payload, value);
            }
        }
        return payload;
    }

    std::vector<uint8_t> prototype_metadata_payload() const {
        std::vector<uint8_t> payload;
        payload.reserve(prototypes.size() * 64);
        for (const prototype & slot : prototypes) {
            payload.push_back(slot.active ? 1 : 0);
            payload.insert(payload.end(), 7, 0);
            append_u64(payload, slot.count);
            append_double(payload, slot.mean_coefficient);
            append_double(payload, slot.key_dispersion);
            append_double(payload, slot.q_coherence);
            append_u64(payload, slot.creation_step);
            append_u64(payload, slot.last_update_step);
            append_u64(payload, slot.forced_merge_count);
        }
        return payload;
    }

    std::vector<uint8_t> prototype_state_payload() const {
        std::vector<uint8_t> payload;
        payload.reserve(5 * sizeof(uint64_t));
        append_u64(payload, writes);
        append_u64(payload, lifetime_merges);
        append_u64(payload, forced_merges);
        append_u64(payload, total_created_prototypes);
        append_u64(payload, experiences_absorbed);
        return payload;
    }

    void mark_prototype_dirty() {
        prototype_hash_dirty = true;
        prototype_hash_cache.clear();
    }

    prototype_update absorb_prototype(
            const std::vector<float> & incoming_key,
            const std::vector<float> & incoming_q,
            double coefficient) {
        if (!is_prototype_backend()) {
            throw std::runtime_error("prototype update requested for raw sidecar");
        }
        if (incoming_key.size() != size_t(width) || incoming_q.size() != size_t(width)) {
            throw std::runtime_error("prototype update width mismatch");
        }
        std::vector<float> key = incoming_key;
        normalize_vector(key);
        if (vector_norm(key) <= EPSILON) {
            throw std::runtime_error("prototype key has zero norm");
        }

        size_t best_index = prototypes.size();
        double best_similarity = -std::numeric_limits<double>::infinity();
        for (size_t index = 0; index < prototypes.size(); ++index) {
            const prototype & slot = prototypes[index];
            if (!slot.active) {
                continue;
            }
            const double similarity = cosine_similarity(slot.centroid_key, key);
            if (similarity > best_similarity ||
                    (close_enough(similarity, best_similarity) && index < best_index)) {
                best_index = index;
                best_similarity = similarity;
            }
        }

        size_t inactive_index = prototypes.size();
        for (size_t index = 0; index < prototypes.size(); ++index) {
            if (!prototypes[index].active) {
                inactive_index = index;
                break;
            }
        }

        const bool create = inactive_index < prototypes.size() &&
            (best_index == prototypes.size() || best_similarity < merge_threshold);
        if (create) {
            prototype & slot = prototypes[inactive_index];
            slot.active = true;
            slot.count = 1;
            slot.mean_coefficient = coefficient;
            slot.key_dispersion = 0.0;
            slot.q_coherence = 1.0;
            slot.creation_step = experiences_absorbed + 1;
            slot.last_update_step = experiences_absorbed + 1;
            slot.forced_merge_count = 0;
            slot.centroid_key = key;
            slot.mean_q = incoming_q;
            ++total_created_prototypes;
            ++experiences_absorbed;
            mark_prototype_dirty();
            return prototype_update{inactive_index, 0.0, true, false};
        }

        if (best_index == prototypes.size()) {
            throw std::runtime_error("prototype bank has no usable slot");
        }
        prototype & slot = prototypes[best_index];
        const uint64_t previous_count = slot.count;
        const double denominator = double(previous_count + 1);
        const double q_similarity = cosine_similarity(slot.mean_q, incoming_q);
        for (size_t i = 0; i < size_t(width); ++i) {
            slot.centroid_key[i] = float(
                double(slot.centroid_key[i]) +
                (double(key[i]) - double(slot.centroid_key[i])) / denominator);
            slot.mean_q[i] = float(
                double(slot.mean_q[i]) +
                (double(incoming_q[i]) - double(slot.mean_q[i])) / denominator);
        }
        normalize_vector(slot.centroid_key);
        slot.mean_coefficient += (coefficient - slot.mean_coefficient) / denominator;
        slot.key_dispersion += ((1.0 - best_similarity) - slot.key_dispersion) / denominator;
        slot.q_coherence += (q_similarity - slot.q_coherence) / denominator;
        slot.count = previous_count + 1;
        slot.last_update_step = experiences_absorbed + 1;
        ++lifetime_merges;
        ++experiences_absorbed;

        const bool forced = active_prototype_count() == prototypes.size() &&
            best_similarity < merge_threshold;
        if (forced) {
            ++slot.forced_merge_count;
            ++forced_merges;
        }
        mark_prototype_dirty();
        return prototype_update{best_index, best_similarity, false, forced};
    }

    void append_component(component value) {
        const std::string metadata = value.source + "\n" + value.destination + "\n" +
            std::to_string(value.coefficient) + "\n";
        sha256_update(
            state_hash_context,
            reinterpret_cast<const uint8_t *>(metadata.data()),
            metadata.size());
        sha256_update(
            state_hash_context,
            reinterpret_cast<const uint8_t *>(value.value.data()),
            value.value.size() * sizeof(float));
        sha256_update(
            state_hash_context,
            reinterpret_cast<const uint8_t *>(value.key.data()),
            value.key.size() * sizeof(float));
        components.push_back(std::move(value));
    }

    void clear_components() {
        components.clear();
        state_hash_context = sha256_context{};
    }

    std::vector<float> retrieve(
            const float * query,
            double & raw_norm,
            double & budget,
            std::string & source) const {
        std::vector<float> response(width, 0.0f);
        if (is_prototype_backend()) {
            std::vector<std::pair<double, size_t>> candidates;
            candidates.reserve(active_prototype_count());
            for (size_t index = 0; index < prototypes.size(); ++index) {
                const prototype & slot = prototypes[index];
                if (slot.active) {
                    candidates.emplace_back(
                        cosine_similarity(slot.centroid_key, query, size_t(width)), index);
                }
            }
            std::sort(candidates.begin(), candidates.end(), [](const auto & left, const auto & right) {
                if (!close_enough(left.first, right.first)) {
                    return left.first > right.first;
                }
                return left.second < right.second;
            });
            const size_t limit = std::min(
                candidates.size(), size_t(std::max(1, top_cluster_candidates)));
            double selected_activation = 0.0;
            double selected_abs_activation = -1.0;
            for (size_t rank = 0; rank < limit; ++rank) {
                const double activation = candidates[rank].first;
                const prototype & slot = prototypes[candidates[rank].second];
                if (std::abs(activation) > selected_abs_activation) {
                    selected_abs_activation = std::abs(activation);
                    selected_activation = activation;
                    source = "prototype:" + std::to_string(candidates[rank].second);
                }
                for (int32_t i = 0; i < width; ++i) {
                    response[size_t(i)] += float(
                        slot.mean_coefficient * activation * double(slot.mean_q[size_t(i)]));
                }
            }
            raw_norm = vector_norm(response);
            budget = server_sidecar_authority_controller::budget(raw_norm);
            if (selected_abs_activation < 0.0) {
                source.clear();
            }
            (void) selected_activation;
            return response;
        }
        double selected_activation = 0.0;
        double selected_abs_activation = -1.0;
        for (const component & component : components) {
            const double activation = dot_product(component.key, query, size_t(width));
            if (std::abs(activation) > selected_abs_activation) {
                selected_abs_activation = std::abs(activation);
                selected_activation = activation;
                source = component.source;
            }
            for (int32_t i = 0; i < width; ++i) {
                response[size_t(i)] += float(component.coefficient * activation * double(component.value[size_t(i)]));
            }
        }
        raw_norm = vector_norm(response);
        budget = server_sidecar_authority_controller::budget(raw_norm);
        if (selected_abs_activation < 0.0) {
            source.clear();
        }
        (void) selected_activation;
        return response;
    }

    // Research-only single-candidate read.  The normal reader first computes
    // the aggregate Top-K norm and authority.  The selected prototype is then
    // rescaled to that same effective read magnitude, matching the causal
    // utility probe's single-candidate intervention while leaving the stored
    // key/value state unchanged.
    std::vector<float> retrieve_forced(
            const float * query,
            size_t prototype_index,
            double target_raw_norm,
            double & raw_norm,
            double & budget,
            std::string & source,
            std::string & error) const {
        if (!is_prototype_backend()) {
            error = "forced prototype selection requires the prototype backend";
            return {};
        }
        if (prototype_index >= prototypes.size() || !prototypes[prototype_index].active) {
            error = "forced prototype_id is not an active prototype";
            return {};
        }

        const prototype & slot = prototypes[prototype_index];
        const double activation = cosine_similarity(
            slot.centroid_key, query, size_t(width));
        const double q_norm = vector_norm(slot.mean_q);
        std::vector<float> response(width, 0.0f);
        if (target_raw_norm <= EPSILON) {
            error = "forced prototype read has zero normal target magnitude";
            return {};
        }
        if (std::abs(activation) <= EPSILON) {
            error = "forced prototype read has zero query activation";
            return {};
        }
        if (q_norm <= EPSILON) {
            error = "forced prototype read has zero correction norm";
            return {};
        }
        const double sign = slot.mean_coefficient * activation < 0.0 ? -1.0 : 1.0;
        const double coefficient = sign * target_raw_norm / (activation * q_norm);
        for (int32_t i = 0; i < width; ++i) {
            response[size_t(i)] = float(
                coefficient * activation * double(slot.mean_q[size_t(i)]));
        }
        raw_norm = vector_norm(response);
        budget = server_sidecar_authority_controller::budget(raw_norm);
        source = "prototype:" + std::to_string(prototype_index);
        return response;
    }

    std::string state_hash() const {
        if (is_prototype_backend()) {
            if (!prototype_hash_dirty && !prototype_hash_cache.empty()) {
                return prototype_hash_cache;
            }
            sha256_context context;
            const std::string config = "sidecar_prototype_v1\n" +
                std::to_string(width) + "\n" +
                std::to_string(configured_prototypes) + "\n" +
                std::to_string(merge_threshold) + "\n" +
                std::to_string(top_cluster_candidates) + "\n";
            sha256_update(context, reinterpret_cast<const uint8_t *>(config.data()), config.size());
            const auto vectors = prototype_vectors_payload();
            const auto metadata = prototype_metadata_payload();
            const auto state = prototype_state_payload();
            sha256_update(context, vectors.data(), vectors.size());
            sha256_update(context, metadata.data(), metadata.size());
            sha256_update(context, state.data(), state.size());
            prototype_hash_cache = sha256_finish(context);
            prototype_hash_dirty = false;
            return prototype_hash_cache;
        }
        sha256_context snapshot = state_hash_context;
        return sha256_finish(snapshot);
    }
};

void server_sidecar_manager::configure(llama_model * new_model, const std::string & sidecar_root) {
    std::lock_guard<std::mutex> lock(mutex);
    if (model != new_model) {
        // Sidecars are tied to the loaded model dimensions and runtime.  Do
        // not allow state from a previous model/context to survive reload.
        sidecars.clear();
    }
    model = new_model;
    model_width = model ? llama_model_n_embd(model) : 0;
    model_layers = model ? llama_model_n_layer(model) : 0;
    root.clear();
    if (!sidecar_root.empty()) {
        root = fs::absolute(fs::path(sidecar_root)).lexically_normal().string();
    }
    injection_profile = nullptr;
    profile_error.clear();
    if (model != nullptr) {
        injection_profile = server_sidecar_profile_registry::for_model(model, {}, profile_error);
    }
    std::error_code error;
    if (!root.empty()) {
        fs::create_directories(root, error);
        if (error) {
            throw std::runtime_error("cannot create sidecar directory '" + root + "': " + error.message());
        }
    }
    for (const auto & entry : sidecars) {
        std::lock_guard<std::mutex> sidecar_lock(entry.second->mutex);
        entry.second->episodes.clear();
    }
}

void server_sidecar_manager::clear_context(llama_context * ctx) const {
    if (ctx == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (model_width > 0) {
        llama_set_adapter_cvec(ctx, nullptr, 0, model_width, -1, -1);
    }
    llama_clear_step30_intervention(ctx);
}

void server_sidecar_manager::invalidate_episodes() {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & entry : sidecars) {
        std::lock_guard<std::mutex> sidecar_lock(entry.second->mutex);
        entry.second->episodes.clear();
    }
}

bool server_sidecar_manager::model_ready() const {
    std::lock_guard<std::mutex> lock(mutex);
    return model != nullptr && model_width > 0;
}

std::shared_ptr<server_sidecar_manager::sidecar> server_sidecar_manager::get(const std::string & id) const {
    auto it = sidecars.find(id);
    if (it == sidecars.end()) {
        throw std::runtime_error("unknown sidecar_id: " + id);
    }
    return it->second;
}

bool server_sidecar_manager::has(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mutex);
    return sidecars.find(id) != sidecars.end();
}

bool server_sidecar_manager::capture_query(
        const std::string & id,
        llama_context * ctx,
        int32_t output_index,
        int32_t width,
        std::vector<float> & query,
        std::string & error) {
    if (ctx == nullptr) {
        error = "sidecar query context is unavailable";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (injection_profile == nullptr) {
        error = profile_error.empty() ?
            "sidecar_injection_profile_not_supported" : profile_error;
        return false;
    }
    if (injection_profile->query_hook != server_sidecar_hook::result_norm) {
        error = "sidecar query hook is not supported by this backend";
        return false;
    }
    if (model_width != width) {
        error = "sidecar query width does not match model width";
        return false;
    }
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    const float * captured = llama_get_embeddings_ith(ctx, output_index);
    if (captured == nullptr) {
        error = "sidecar query embedding is unavailable";
        return false;
    }
    query.assign(captured, captured + width);
    ++state->query_captures;
    return true;
}

std::string server_sidecar_manager::package_path(const std::string & path, bool for_write) const {
    if (root.empty()) {
        throw std::runtime_error("sidecar directory is not configured");
    }
    const fs::path root_path = fs::absolute(fs::path(root)).lexically_normal();
    const fs::path requested = path.empty() ? root_path :
        (fs::path(path).is_absolute() ? fs::path(path) : root_path / fs::path(path));
    const fs::path normalized = requested.lexically_normal();
    const fs::path relative = normalized.lexically_relative(root_path);
    if (relative.empty() || relative == "." || relative.string().rfind("..", 0) == 0) {
        throw std::runtime_error("sidecar path is outside the configured sidecar directory");
    }
    if (!for_write && !fs::is_directory(normalized)) {
        throw std::runtime_error("sidecar package is not a directory: " + normalized.string());
    }
    return normalized.string();
}

json server_sidecar_manager::status(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (id.empty()) {
        json result = json::object();
        result["enabled"] = !root.empty();
        result["sidecars"] = json::array();
        for (const auto & entry : sidecars) {
            const auto & state = *entry.second;
            std::lock_guard<std::mutex> sidecar_lock(state.mutex);
            result["sidecars"].push_back(json{
                {"id", state.id},
                {"loaded", state.loaded},
                {"format_version", state.is_prototype_backend() ? 2 : 1},
                {"memory_backend", state.memory_backend},
                {"learning_enabled", state.learning_enabled},
                {"sha256", state.state_hash()},
                {"components", state.components.size()},
                {"configured_prototypes", state.configured_prototypes},
                {"allocated_prototypes", state.prototypes.size()},
                {"active_prototypes", state.active_prototype_count()},
                {"prototype_count", state.active_prototype_count()},
                {"lifetime_writes", state.writes},
                {"lifetime_merges", state.lifetime_merges},
                {"forced_merges", state.forced_merges},
                {"total_created_prototypes", state.total_created_prototypes},
                {"experiences_absorbed", state.experiences_absorbed},
                {"merge_threshold", state.merge_threshold},
                {"top_cluster_candidates", state.top_cluster_candidates},
                {"width", state.width},
                {"bytes", state.is_prototype_backend() ? state.serialized_file_bytes() :
                    state.components.size() * size_t(state.width) * sizeof(float) * 2},
                {"writes", state.writes},
                {"retrievals", state.retrievals},
                {"mean_B", state.retrievals == 0 ? 0.0 : state.sum_b / double(state.retrievals)},
                {"max_B", state.max_b},
                {"cumulative_B", state.sum_b},
                {"injection_profile_id", state.injection_profile_id},
                {"profile_supported", injection_profile != nullptr},
                {"profile_error", profile_error},
                {"query_hook", "result_norm"},
                {"supported_injection_sites", injection_profile == nullptr ? 0 : injection_profile->supported_sites.size()},
                {"research_injection_sites", injection_profile == nullptr ? 0 : injection_profile->research_sites.size()},
                {"recursive_state_path_supported", injection_profile != nullptr && injection_profile->recursive_state_path_supported},
                {"query_captures", state.query_captures},
                {"injection_site_dispatches", state.dispatch_telemetry.injection_sites},
                {"injection_site_24", state.dispatch_telemetry.graph_layer_24},
                {"injection_site_28", state.dispatch_telemetry.graph_layer_28},
                {"injection_mode", "validated_step30_runtime"},
                {"final_logit_injection_supported", injection_profile != nullptr && injection_profile->final_logit_injection_supported},
            });
        }
        return result;
    }

    const auto state = get(id);
    std::lock_guard<std::mutex> sidecar_lock(state->mutex);
    return json{
        {"enabled", !root.empty()},
        {"id", state->id},
        {"loaded", state->loaded},
        {"format_version", state->is_prototype_backend() ? 2 : 1},
        {"memory_backend", state->memory_backend},
        {"learning_enabled", state->learning_enabled},
        {"sha256", state->state_hash()},
        {"components", state->components.size()},
        {"configured_prototypes", state->configured_prototypes},
        {"allocated_prototypes", state->prototypes.size()},
        {"active_prototypes", state->active_prototype_count()},
        {"prototype_count", state->active_prototype_count()},
        {"lifetime_writes", state->writes},
        {"lifetime_merges", state->lifetime_merges},
        {"forced_merges", state->forced_merges},
        {"total_created_prototypes", state->total_created_prototypes},
        {"experiences_absorbed", state->experiences_absorbed},
        {"merge_threshold", state->merge_threshold},
        {"top_cluster_candidates", state->top_cluster_candidates},
        {"width", state->width},
        {"bytes", state->is_prototype_backend() ? state->serialized_file_bytes() :
            state->components.size() * size_t(state->width) * sizeof(float) * 2},
        {"writes", state->writes},
        {"retrievals", state->retrievals},
        {"mean_B", state->retrievals == 0 ? 0.0 : state->sum_b / double(state->retrievals)},
        {"max_B", state->max_b},
        {"cumulative_B", state->sum_b},
        {"injection_profile_id", state->injection_profile_id},
        {"profile_supported", injection_profile != nullptr},
        {"profile_error", profile_error},
        {"query_hook", "result_norm"},
        {"supported_injection_sites", injection_profile == nullptr ? 0 : injection_profile->supported_sites.size()},
        {"research_injection_sites", injection_profile == nullptr ? 0 : injection_profile->research_sites.size()},
        {"recursive_state_path_supported", injection_profile != nullptr && injection_profile->recursive_state_path_supported},
        {"query_captures", state->query_captures},
        {"injection_site_dispatches", state->dispatch_telemetry.injection_sites},
        {"injection_site_24", state->dispatch_telemetry.graph_layer_24},
        {"injection_site_28", state->dispatch_telemetry.graph_layer_28},
        {"injection_mode", "validated_step30_runtime"},
        {"final_logit_injection_supported", injection_profile != nullptr && injection_profile->final_logit_injection_supported},
    };
}

json server_sidecar_manager::create(
        const std::string & id,
        bool learning_enabled,
        const std::string & memory_backend,
        int32_t configured_prototypes,
        double merge_threshold,
        int32_t top_cluster_candidates) {
    if (!valid_id(id)) {
        throw std::runtime_error("invalid sidecar_id");
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (root.empty()) {
        throw std::runtime_error("sidecar integration is disabled; pass --sidecar-dir");
    }
    if (!model || model_width <= 0) {
        throw std::runtime_error("sidecar manager is not bound to a model");
    }
    if (injection_profile == nullptr) {
        throw std::runtime_error(profile_error.empty() ?
            "sidecar_injection_profile_not_supported" : profile_error);
    }
    if (sidecars.find(id) != sidecars.end()) {
        throw std::runtime_error("sidecar_id already exists: " + id);
    }
    if (memory_backend != "raw" && memory_backend != "prototype") {
        throw std::runtime_error("invalid sidecar memory backend: " + memory_backend);
    }
    if (configured_prototypes < 1) {
        throw std::runtime_error("prototype capacity must be positive");
    }
    if (!std::isfinite(merge_threshold) || merge_threshold < -1.0 || merge_threshold > 1.0) {
        throw std::runtime_error("prototype merge threshold must be in [-1, 1]");
    }
    if (top_cluster_candidates < 1) {
        throw std::runtime_error("prototype candidate count must be positive");
    }
    auto state = std::make_shared<sidecar>();
    state->id = id;
    state->width = model_width;
    state->learning_enabled = learning_enabled;
    state->memory_backend = memory_backend;
    state->merge_threshold = merge_threshold;
    state->top_cluster_candidates = top_cluster_candidates;
    state->injection_profile_id = injection_profile->id;
    if (memory_backend == "prototype") {
        state->initialize_prototype_slots(configured_prototypes);
    }
    sidecars.emplace(id, state);
    return json{
        {"id", id},
        {"loaded", true},
        {"learning_enabled", learning_enabled},
        {"memory_backend", memory_backend},
        {"configured_prototypes", memory_backend == "prototype" ? configured_prototypes : 0},
        {"active_prototypes", 0},
        {"components", 0},
        {"width", model_width},
    };
}

json server_sidecar_manager::clone(const std::string & source_id, const std::string & clone_id, bool learning_enabled) {
    if (!valid_id(clone_id)) {
        throw std::runtime_error("invalid sidecar_id");
    }
    if (source_id == clone_id) {
        throw std::runtime_error("source and clone sidecar IDs must differ");
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (root.empty()) {
        throw std::runtime_error("sidecar integration is disabled; pass --sidecar-dir");
    }
    if (sidecars.find(clone_id) != sidecars.end()) {
        throw std::runtime_error("sidecar_id already exists: " + clone_id);
    }

    const auto source = get(source_id);
    std::lock_guard<std::mutex> source_lock(source->mutex);
    auto state = std::make_shared<sidecar>();
    state->id = clone_id;
    state->width = source->width;
    state->learning_enabled = learning_enabled;
    state->loaded = source->loaded;
    state->writer_counts = source->writer_counts;
    state->next_episode = source->next_episode;
    state->version = source->version;
    state->loaded_path = source->loaded_path;
    state->package_sha = source->package_sha;
    state->base_model_identifier = source->base_model_identifier;
    state->base_model_sha256 = source->base_model_sha256;
    state->injection_profile_id = source->injection_profile_id;
    state->memory_backend = source->memory_backend;
    state->configured_prototypes = source->configured_prototypes;
    state->merge_threshold = source->merge_threshold;
    state->top_cluster_candidates = source->top_cluster_candidates;
    state->lifetime_merges = source->lifetime_merges;
    state->forced_merges = source->forced_merges;
    state->total_created_prototypes = source->total_created_prototypes;
    state->experiences_absorbed = source->experiences_absorbed;
    state->components.reserve(source->components.size());
    for (const auto & component : source->components) {
        state->append_component(component);
    }
    if (source->is_prototype_backend()) {
        state->initialize_prototype_slots(source->configured_prototypes);
        state->prototypes = source->prototypes;
        state->prototype_hash_dirty = true;
    }
    sidecars.emplace(clone_id, state);

    return json{
        {"id", clone_id},
        {"source_id", source_id},
        {"loaded", state->loaded},
        {"learning_enabled", state->learning_enabled},
        {"memory_backend", state->memory_backend},
        {"configured_prototypes", state->configured_prototypes},
        {"active_prototypes", state->active_prototype_count()},
        {"components", state->components.size()},
        {"width", state->width},
        {"sha256", state->state_hash()},
    };
}

json server_sidecar_manager::load(const std::string & id, const std::string & path, bool learning_enabled) {
    if (!valid_id(id)) {
        throw std::runtime_error("invalid sidecar_id");
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (injection_profile == nullptr) {
            throw std::runtime_error(profile_error.empty() ?
                "sidecar_injection_profile_not_supported" : profile_error);
        }
    }
    const fs::path package = package_path(path, false);
    const fs::path manifest_path = package / "manifest.json";
    const auto manifest_bytes = read_bytes(manifest_path);
    const json manifest = json::parse(std::string(manifest_bytes.begin(), manifest_bytes.end()));

    if (manifest.value("magic", std::string()) == "sidecar_prototype_v1") {
        require_json_file(manifest, "backend", "prototype");
        if (required_json<int>(manifest, "version") != 1 ||
                required_json<std::string>(manifest, "dtype") != "float32") {
            throw std::runtime_error("unsupported prototype sidecar version or dtype");
        }
        if (!manifest.contains("shape") || !manifest["shape"].is_object()) {
            throw std::runtime_error("prototype sidecar manifest has no shape");
        }
        const int32_t width = required_json<int32_t>(manifest["shape"], "width");
        const int32_t configured_prototypes = required_json<int32_t>(
            manifest["shape"], "prototype_count");
        if (width <= 0 || configured_prototypes < 1 ||
                required_json<int32_t>(manifest, "configured_prototypes") != configured_prototypes) {
            throw std::runtime_error("prototype sidecar manifest shape is inconsistent");
        }
        if (model_width > 0 && width != model_width) {
            throw std::runtime_error("prototype sidecar width does not match loaded model width");
        }
        const double merge_threshold = required_json<double>(manifest, "merge_threshold");
        const int32_t top_cluster_candidates = required_json<int32_t>(
            manifest, "top_cluster_candidates");
        if (!std::isfinite(merge_threshold) || merge_threshold < -1.0 || merge_threshold > 1.0 ||
                top_cluster_candidates < 1) {
            throw std::runtime_error("invalid prototype sidecar configuration");
        }

        const fs::path vectors_path = package / "prototypes.f32";
        const fs::path metadata_path = package / "prototypes.meta";
        const fs::path state_path = package / "state.meta";
        const auto vectors_bytes = read_bytes(vectors_path);
        const auto metadata_bytes = read_bytes(metadata_path);
        const auto state_bytes = read_bytes(state_path);
        const size_t expected_vectors = size_t(configured_prototypes) * size_t(width) * 2 * sizeof(float);
        const size_t expected_metadata = size_t(configured_prototypes) * 64;
        if (vectors_bytes.size() != expected_vectors || metadata_bytes.size() != expected_metadata ||
                state_bytes.size() != 5 * sizeof(uint64_t)) {
            throw std::runtime_error("prototype sidecar fixed payload size mismatch");
        }
        if (sha256_bytes(vectors_bytes) != required_json<std::string>(manifest, "payload_sha256") ||
                sha256_bytes(metadata_bytes) != required_json<std::string>(manifest, "metadata_sha256") ||
                sha256_bytes(state_bytes) != required_json<std::string>(manifest, "state_sha256")) {
            throw std::runtime_error("prototype sidecar checksum mismatch");
        }

        auto state = std::make_shared<sidecar>();
        state->id = id;
        state->width = width;
        state->learning_enabled = learning_enabled;
        state->memory_backend = "prototype";
        state->merge_threshold = merge_threshold;
        state->top_cluster_candidates = top_cluster_candidates;
        state->injection_profile_id = injection_profile->id;
        state->loaded_path = package.string();
        state->package_sha = sha256_bytes(manifest_bytes) + sha256_bytes(metadata_bytes) +
            sha256_bytes(state_bytes) + sha256_bytes(vectors_bytes);
        if (manifest.contains("base_model") && manifest["base_model"].is_object()) {
            state->base_model_identifier = manifest["base_model"].value("identifier", "unavailable");
            state->base_model_sha256 = manifest["base_model"].value("sha256", "");
        }
        state->initialize_prototype_slots(configured_prototypes);

        size_t vector_offset = 0;
        for (sidecar::prototype & slot : state->prototypes) {
            for (float & value : slot.centroid_key) {
                value = read_float(vectors_bytes, vector_offset);
            }
            for (float & value : slot.mean_q) {
                value = read_float(vectors_bytes, vector_offset);
            }
        }
        if (vector_offset != vectors_bytes.size()) {
            throw std::runtime_error("prototype vector payload has trailing bytes");
        }

        size_t metadata_offset = 0;
        for (sidecar::prototype & slot : state->prototypes) {
            if (metadata_offset + 8 > metadata_bytes.size()) {
                throw std::runtime_error("truncated prototype slot metadata");
            }
            slot.active = metadata_bytes[metadata_offset++] != 0;
            metadata_offset += 7;
            slot.count = read_u64(metadata_bytes, metadata_offset);
            slot.mean_coefficient = read_double(metadata_bytes, metadata_offset);
            slot.key_dispersion = read_double(metadata_bytes, metadata_offset);
            slot.q_coherence = read_double(metadata_bytes, metadata_offset);
            slot.creation_step = read_u64(metadata_bytes, metadata_offset);
            slot.last_update_step = read_u64(metadata_bytes, metadata_offset);
            slot.forced_merge_count = read_u64(metadata_bytes, metadata_offset);
            if (!slot.active && slot.count != 0) {
                throw std::runtime_error("inactive prototype slot has a nonzero count");
            }
        }
        if (metadata_offset != metadata_bytes.size()) {
            throw std::runtime_error("prototype metadata has trailing bytes");
        }

        size_t state_offset = 0;
        state->writes = read_u64(state_bytes, state_offset);
        state->lifetime_merges = read_u64(state_bytes, state_offset);
        state->forced_merges = read_u64(state_bytes, state_offset);
        state->total_created_prototypes = read_u64(state_bytes, state_offset);
        state->experiences_absorbed = read_u64(state_bytes, state_offset);
        if (state_offset != state_bytes.size()) {
            throw std::runtime_error("prototype state metadata has trailing bytes");
        }
        state->prototype_hash_dirty = true;

        std::lock_guard<std::mutex> lock(mutex);
        if (sidecars.find(id) != sidecars.end()) {
            throw std::runtime_error("sidecar_id already exists: " + id);
        }
        sidecars[id] = std::move(state);
        const auto loaded = sidecars.at(id);
        std::lock_guard<std::mutex> state_lock(loaded->mutex);
        return json{
            {"id", id},
            {"loaded", true},
            {"learning_enabled", learning_enabled},
            {"memory_backend", "prototype"},
            {"configured_prototypes", loaded->configured_prototypes},
            {"active_prototypes", loaded->active_prototype_count()},
            {"components", 0},
            {"width", loaded->width},
            {"sha256", loaded->state_hash()},
        };
    }

    const fs::path components_path = package / "components.jsonl";
    const fs::path vectors_path = package / "vectors.f32";

    const auto components_bytes = read_bytes(components_path);
    const auto vectors_bytes = read_bytes(vectors_path);
    require_json_file(manifest, "magic", "sidecar_v1");
    if (required_json<int>(manifest, "version") != 1 || required_json<std::string>(manifest, "dtype") != "float32") {
        throw std::runtime_error("unsupported sidecar_v1 version or dtype");
    }
    if (!manifest.contains("shape") || !manifest["shape"].is_object()) {
        throw std::runtime_error("sidecar manifest has no shape");
    }
    const int32_t width = required_json<int32_t>(manifest["shape"], "width");
    const size_t component_count = required_json<size_t>(manifest["shape"], "component_count");
    if (width <= 0 || component_count != required_json<size_t>(manifest, "component_count")) {
        throw std::runtime_error("sidecar manifest shape is inconsistent");
    }
    if (model_width > 0 && width != model_width) {
        throw std::runtime_error("sidecar width does not match loaded model width");
    }
    if (sha256_bytes(vectors_bytes) != required_json<std::string>(manifest, "payload_sha256")) {
        throw std::runtime_error("sidecar payload checksum mismatch");
    }
    if (sha256_bytes(components_bytes) != required_json<std::string>(manifest, "components_sha256")) {
        throw std::runtime_error("sidecar component checksum mismatch");
    }
    if (vectors_bytes.size() != component_count * size_t(width) * 2 * sizeof(float)) {
        throw std::runtime_error("sidecar vector payload size mismatch");
    }
    if (manifest.contains("reader")) {
        const json & reader = manifest.at("reader");
        if (reader.contains("b_max") && !close_enough(
                reader.at("b_max").get<double>(),
                server_sidecar_authority_controller::max_budget())) {
            throw std::runtime_error("sidecar authority does not match validated B_max");
        }
        if (reader.contains("s_ref") && !close_enough(
                reader.at("s_ref").get<double>(),
                server_sidecar_authority_controller::reference_norm())) {
            throw std::runtime_error("sidecar reader does not match validated s_ref");
        }
        if (reader.contains("topology")) {
            const json & topology = reader.at("topology");
            if (!close_enough(
                    topology.value("block24", server_sidecar_validated_topology::block24_scale()),
                    server_sidecar_validated_topology::block24_scale()) ||
                !close_enough(
                    topology.value("block28", server_sidecar_validated_topology::block28_scale()),
                    server_sidecar_validated_topology::block28_scale()) ||
                !close_enough(
                    topology.value("final", server_sidecar_validated_topology::final_scale()),
                    server_sidecar_validated_topology::final_scale())) {
                throw std::runtime_error("sidecar injection topology does not match validated constants");
            }
        }
    }

    auto state = std::make_shared<sidecar>();
    state->id = id;
    state->width = width;
    state->learning_enabled = learning_enabled;
    state->injection_profile_id = injection_profile->id;
    state->loaded_path = package.string();
    state->package_sha = sha256_bytes(manifest_bytes) + sha256_bytes(components_bytes) + sha256_bytes(vectors_bytes);
    state->components.reserve(component_count);
    if (manifest.contains("base_model") && manifest["base_model"].is_object()) {
        state->base_model_identifier = manifest["base_model"].value("identifier", "unavailable");
        state->base_model_sha256 = manifest["base_model"].value("sha256", "");
    }

    std::istringstream components_stream(std::string(components_bytes.begin(), components_bytes.end()));
    std::string line;
    size_t component_index = 0;
    while (std::getline(components_stream, line)) {
        if (line.empty()) {
            continue;
        }
        const json item = json::parse(line);
        const size_t index = required_json<size_t>(item, "component");
        if (index != component_index) {
            throw std::runtime_error("sidecar components are not ordered");
        }
        sidecar::component component;
        component.coefficient = required_json<double>(item, "coefficient");
        component.source = required_json<std::string>(item, "source");
        component.destination = required_json<std::string>(item, "destination");
        component.value.resize(size_t(width));
        component.key.resize(size_t(width));
        const size_t offset = component_index * size_t(width) * 2 * sizeof(float);
        std::memcpy(component.key.data(), vectors_bytes.data() + offset, size_t(width) * sizeof(float));
        std::memcpy(component.value.data(), vectors_bytes.data() + offset + size_t(width) * sizeof(float), size_t(width) * sizeof(float));
        if (item.contains("value_checksum") && item.at("value_checksum").get<std::string>() != vector_checksum(component.value)) {
            throw std::runtime_error("sidecar value checksum mismatch");
        }
        if (item.contains("key_checksum") && item.at("key_checksum").get<std::string>() != vector_checksum(component.key)) {
            throw std::runtime_error("sidecar key checksum mismatch");
        }
        state->append_component(std::move(component));
        ++component_index;
    }
    if (component_index != component_count) {
        throw std::runtime_error("sidecar component count mismatch");
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (sidecars.find(id) != sidecars.end()) {
        throw std::runtime_error("sidecar_id already exists: " + id);
    }
    sidecars[id] = std::move(state);
    const auto loaded = sidecars.at(id);
    std::lock_guard<std::mutex> state_lock(loaded->mutex);
    return json{{"id", id}, {"loaded", true}, {"learning_enabled", learning_enabled},
        {"components", loaded->components.size()}, {"width", loaded->width}, {"sha256", loaded->state_hash()}};
}

json server_sidecar_manager::save(const std::string & id, const std::string & path) {
    std::shared_ptr<sidecar> state;
    {
        std::lock_guard<std::mutex> lock(mutex);
        state = get(id);
    }
    std::lock_guard<std::mutex> state_lock(state->mutex);
    const fs::path package = package_path(path.empty() ? id : path, true);
    std::error_code error;
    fs::create_directories(package, error);
    if (error) {
        throw std::runtime_error("cannot create sidecar package: " + error.message());
    }

    if (state->is_prototype_backend()) {
        const auto vectors = state->prototype_vectors_payload();
        const auto metadata = state->prototype_metadata_payload();
        const auto state_metadata = state->prototype_state_payload();
        json manifest = {
            {"magic", "sidecar_prototype_v1"},
            {"version", 1},
            {"backend", "prototype"},
            {"dtype", "float32"},
            {"shape", {{"prototype_count", state->configured_prototypes}, {"width", state->width}}},
            {"configured_prototypes", state->configured_prototypes},
            {"merge_threshold", state->merge_threshold},
            {"top_cluster_candidates", state->top_cluster_candidates},
            {"payload_sha256", sha256_bytes(vectors)},
            {"metadata_sha256", sha256_bytes(metadata)},
            {"state_sha256", sha256_bytes(state_metadata)},
            {"base_model", {{"identifier", state->base_model_identifier}}},
            {"reader", {
                {"b_max", server_sidecar_authority_controller::max_budget()},
                {"s_ref", server_sidecar_authority_controller::reference_norm()},
                {"topology", {
                    {"block24", server_sidecar_validated_topology::block24_scale()},
                    {"block28", server_sidecar_validated_topology::block28_scale()},
                    {"final", server_sidecar_validated_topology::final_scale()},
                }},
                {"implementation", "step29_recursive_c2"},
                {"version", 1},
            }},
        };
        const std::string manifest_text = manifest.dump(2) + "\n";
        write_bytes(package / "prototypes.f32", vectors);
        write_bytes(package / "prototypes.meta", metadata);
        write_bytes(package / "state.meta", state_metadata);
        write_bytes(package / "manifest.json", std::vector<uint8_t>(manifest_text.begin(), manifest_text.end()));
        std::error_code remove_error;
        fs::remove(package / "components.jsonl", remove_error);
        remove_error.clear();
        fs::remove(package / "vectors.f32", remove_error);
        state->loaded_path = package.string();
        state->package_sha = sha256_string(manifest_text) + sha256_bytes(metadata) +
            sha256_bytes(state_metadata) + sha256_bytes(vectors);
        const uintmax_t file_bytes = fs::file_size(package / "manifest.json") +
            fs::file_size(package / "prototypes.f32") + fs::file_size(package / "prototypes.meta") +
            fs::file_size(package / "state.meta");
        return json{
            {"id", state->id},
            {"path", package.string()},
            {"sha256", state->state_hash()},
            {"size", file_bytes},
            {"file_bytes", file_bytes},
            {"memory_backend", "prototype"},
            {"configured_prototypes", state->configured_prototypes},
            {"active_prototypes", state->active_prototype_count()},
            {"components", 0},
        };
    }

    std::vector<uint8_t> vectors;
    vectors.reserve(state->components.size() * size_t(state->width) * 2 * sizeof(float));
    std::string components_text;
    for (size_t index = 0; index < state->components.size(); ++index) {
        const auto & component = state->components[index];
        const auto * key_begin = reinterpret_cast<const uint8_t *>(component.key.data());
        vectors.insert(vectors.end(), key_begin, key_begin + component.key.size() * sizeof(float));
        const auto * value_begin = reinterpret_cast<const uint8_t *>(component.value.data());
        vectors.insert(vectors.end(), value_begin, value_begin + component.value.size() * sizeof(float));
        const json item = json{
            {"coefficient", component.coefficient},
            {"component", index},
            {"destination", component.destination},
            {"key_checksum", vector_checksum(component.key)},
            {"key_norm", vector_norm(component.key)},
            {"source", component.source},
            {"value_checksum", vector_checksum(component.value)},
            {"value_norm", vector_norm(component.value)},
        };
        components_text += item.dump();
        components_text += '\n';
    }
    const std::string payload_sha = sha256_bytes(vectors);
    const std::string components_sha = sha256_string(components_text);
    json manifest = {
        {"magic", "sidecar_v1"},
        {"version", 1},
        {"dtype", "float32"},
        {"shape", {{"component_count", state->components.size()}, {"width", state->width}}},
        {"component_count", state->components.size()},
        {"payload_sha256", payload_sha},
        {"components_sha256", components_sha},
        {"base_model", { {"identifier", state->base_model_identifier} }},
        {"reader", {
            {"b_max", server_sidecar_authority_controller::max_budget()},
            {"s_ref", server_sidecar_authority_controller::reference_norm()},
            {"topology", {
                {"block24", server_sidecar_validated_topology::block24_scale()},
                {"block28", server_sidecar_validated_topology::block28_scale()},
                {"final", server_sidecar_validated_topology::final_scale()},
            }},
            {"implementation", "step29_recursive_c2"},
            {"version", 1},
        }},
    };
    const std::string manifest_text = manifest.dump(2) + "\n";
    write_bytes(package / "vectors.f32", vectors);
    write_bytes(package / "components.jsonl", std::vector<uint8_t>(components_text.begin(), components_text.end()));
    write_bytes(package / "manifest.json", std::vector<uint8_t>(manifest_text.begin(), manifest_text.end()));
    std::error_code remove_error;
    fs::remove(package / "prototypes.f32", remove_error);
    remove_error.clear();
    fs::remove(package / "prototypes.meta", remove_error);
    remove_error.clear();
    fs::remove(package / "state.meta", remove_error);
    state->loaded_path = package.string();
    state->package_sha = sha256_string(manifest_text) + components_sha + payload_sha;
    return json{
        {"id", state->id},
        {"path", package.string()},
        {"sha256", state->state_hash()},
        {"size", vectors.size()},
        {"components", state->components.size()},
    };
}

json server_sidecar_manager::reset(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (state->is_prototype_backend()) {
        for (sidecar::prototype & slot : state->prototypes) {
            slot = sidecar::prototype{};
            slot.centroid_key.assign(size_t(state->width), 0.0f);
            slot.mean_q.assign(size_t(state->width), 0.0f);
        }
        state->lifetime_merges = 0;
        state->forced_merges = 0;
        state->total_created_prototypes = 0;
        state->experiences_absorbed = 0;
        state->mark_prototype_dirty();
    } else {
        state->clear_components();
    }
    state->episodes.clear();
    state->writer_counts.clear();
    state->retrievals = 0;
    state->writes = 0;
    state->sum_b = 0.0;
    state->max_b = 0.0;
    state->package_sha.clear();
    ++state->version;
    return json{{"id", id}, {"loaded", true}, {"learning_enabled", state->learning_enabled},
        {"components", state->components.size()}, {"width", state->width}, {"sha256", state->state_hash()}};
}

json server_sidecar_manager::unload(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    state->episodes.clear();
    sidecars.erase(id);
    return json{{"id", id}, {"unloaded", true}};
}

json server_sidecar_manager::erase(const std::string & id) {
    return unload(id);
}

json server_sidecar_manager::set_learning(const std::string & id, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    state->learning_enabled = enabled;
    return json{{"id", id}, {"learning_enabled", enabled}};
}

json server_sidecar_manager::observe(
        const std::string & id,
        const std::string & episode_id,
        llama_token observed_token,
        const std::string & observed_target,
        std::vector<float> * captured_direction,
        std::vector<float> * captured_query) {
    if (captured_direction != nullptr) {
        captured_direction->clear();
    }
    if (captured_query != nullptr) {
        captured_query->clear();
    }
    std::lock_guard<std::mutex> lock(mutex);
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (!state->learning_enabled) {
        const auto episode_it = state->episodes.find(episode_id);
        if (captured_query != nullptr && episode_it != state->episodes.end()) {
            *captured_query = episode_it->second.query;
        }
        if (episode_it != state->episodes.end()) {
            state->episodes.erase(episode_it);
        }
        return json{{"id", id}, {"episode_id", episode_id}, {"updated", false}, {"reason", "learning_disabled"}};
    }
    auto episode_it = state->episodes.find(episode_id);
    if (episode_it == state->episodes.end()) {
        throw std::runtime_error("invalid or expired episode_id");
    }
    if (episode_it->second.version != state->version) {
        throw std::runtime_error("episode invalidated by sidecar state change");
    }
    if (observed_token < 0 || !model ||
            episode_it->second.baseline_greedy < 0) {
        throw std::runtime_error("observe requires a valid single-token episode and observed token");
    }

    std::vector<float> observed_row(size_t(state->width));
    std::vector<float> baseline_row(size_t(state->width));
    if (!llama_model_output_row(model, observed_token, observed_row.data(), observed_row.size()) ||
            !llama_model_output_row(model, episode_it->second.baseline_greedy,
                                    baseline_row.data(), baseline_row.size())) {
        throw std::runtime_error("Step-30 output projection is unavailable for the observed token");
    }

    std::vector<float> direction(observed_row.size());
    for (size_t i = 0; i < direction.size(); ++i) {
        direction[i] = observed_row[i] - baseline_row[i];
    }
    const double direction_norm = vector_norm(direction);
    const std::string source_key = episode_it->second.source;
    const llama_token baseline_greedy = episode_it->second.baseline_greedy;
    std::vector<float> write_key = episode_it->second.query;
    if (captured_query != nullptr) {
        *captured_query = write_key;
    }
    const auto counts_it = state->writer_counts.find(source_key);
    const step30_counts counts_before = counts_it == state->writer_counts.end() ?
        step30_counts{} : counts_it->second;
    const step30_gate gate = compute_step30_gate(counts_before);
    const std::string before = state->state_hash();

    state->episodes.erase(episode_it);
    auto & counts = state->writer_counts[source_key];
    ++counts.total;
    // This is the recurrence/association schedule used by the validated
    // Step-30 runtime.  It is not an accuracy label supplied by the caller.
    ++counts.matching;

    if (!std::isfinite(direction_norm) || direction_norm <= EPSILON) {
        return json{
            {"id", id},
            {"episode_id", episode_id},
            {"updated", false},
            {"reason", "zero_direction"},
            {"observed_token_id", observed_token},
            {"baseline_greedy_token_id", baseline_greedy},
            {"G", gate.gate},
            {"novelty", gate.novelty},
            {"surprise", gate.surprise},
            {"recurrence", gate.recurrence},
            {"consistency", gate.consistency},
            {"write_direction_norm", direction_norm},
            {"sha256_before", before},
            {"sha256_after", state->state_hash()},
            {"components", state->components.size()},
            {"count_total", counts.total},
            {"count_matching", counts.matching},
        };
    }

    sidecar::component component;
    component.coefficient = ETA * gate.gate;
    component.source = source_key;
    component.destination = observed_target.empty() ?
        "token:" + std::to_string(observed_token) : observed_target;
    component.key = std::move(write_key);
    component.value = std::move(direction);
    for (float & value : component.value) {
        value = float(double(value) / direction_norm);
    }
    if (captured_direction != nullptr) {
        *captured_direction = component.value;
    }
    const double update_norm = component.coefficient * vector_norm(component.key);
    const double coefficient = component.coefficient;
    const bool updated = coefficient > 0.0;
    sidecar::prototype_update prototype_result;
    if (updated) {
        if (state->is_prototype_backend()) {
            prototype_result = state->absorb_prototype(
                component.key, component.value, component.coefficient);
        } else {
            state->append_component(std::move(component));
        }
        ++state->writes;
    }
    const std::string after = state->state_hash();
    return json{
        {"id", id},
        {"episode_id", episode_id},
        {"updated", updated},
        {"observed_token_id", observed_token},
        {"baseline_greedy_token_id", baseline_greedy},
        {"G", gate.gate},
        {"novelty", gate.novelty},
        {"surprise", gate.surprise},
        {"recurrence", gate.recurrence},
        {"consistency", gate.consistency},
        {"effective_eta", coefficient},
        {"write_direction_norm", direction_norm},
        {"update_norm", update_norm},
        {"sha256_before", before},
        {"sha256_after", after},
        {"memory_backend", state->memory_backend},
        {"components", state->components.size()},
        {"allocated_prototypes", state->prototypes.size()},
        {"active_prototypes", state->active_prototype_count()},
        {"prototype_index", updated && state->is_prototype_backend() ?
            static_cast<int64_t>(prototype_result.index) : -1},
        {"prototype_created", updated && state->is_prototype_backend() && prototype_result.created},
        {"forced_merge", updated && state->is_prototype_backend() && prototype_result.forced_merge},
        {"count_total", counts.total},
        {"count_matching", counts.matching},
    };
}

json server_sidecar_manager::observe_captured_sequence(
        const std::string & id,
        const std::string & episode_id,
        const std::string & learning_mode,
        int32_t chunk_size,
        const std::vector<server_sidecar_captured_event> & events,
        bool capture_debug_vectors) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);

    if (events.empty()) {
        throw std::runtime_error("captured observation sequence is empty");
    }

    const std::string state_before = state->state_hash();
    const size_t components_before = state->components.size();
    const uint64_t writes_before = state->writes;
    const uint64_t lifetime_merges_before = state->lifetime_merges;
    const uint64_t forced_merges_before = state->forced_merges;
    const uint64_t created_prototypes_before = state->total_created_prototypes;
    const uint64_t experiences_absorbed_before = state->experiences_absorbed;
    std::vector<server_sidecar_manager::sidecar::prototype> prototypes_before;
    bool prototype_hash_dirty_before = true;
    std::string prototype_hash_cache_before;
    if (state->is_prototype_backend()) {
        prototypes_before = state->prototypes;
        prototype_hash_dirty_before = state->prototype_hash_dirty;
        prototype_hash_cache_before = state->prototype_hash_cache;
    }
    if (!state->learning_enabled) {
        return json{
            {"id", id},
            {"episode_id", episode_id},
            {"learning_mode", learning_mode},
            {"chunk_size", chunk_size},
            {"tokens_observed", events.size()},
            {"write_opportunities", events.size()},
            {"writes_realized", 0},
            {"prediction_matches", 0},
            {"components_added", 0},
            {"updated", false},
            {"reason", "learning_disabled"},
            {"sha256_before", state_before},
            {"sha256_after", state_before},
            {"events", json::array()},
        };
    }

    // Appends are rollback-friendly, but writer counters and telemetry also
    // change in sequence. Snapshot those small logical structures so a failed
    // observation cannot leave a half-learned sentence.
    const auto writer_counts_before = state->writer_counts;
    const auto hash_context_before = state->state_hash_context;
    const uint64_t retrievals_before = state->retrievals;
    const double sum_b_before = state->sum_b;
    const double max_b_before = state->max_b;
    const uint64_t query_captures_before = state->query_captures;

    json event_results = json::array();
    size_t writes_realized = 0;
    size_t prediction_matches = 0;
    double gate_sum = 0.0;
    double update_norm_sum = 0.0;
    int32_t previous_position = -1;
    int32_t previous_chunk = -1;
    const bool per_event_hash = !state->is_prototype_backend();

    try {
        for (const auto & event : events) {
            if (event.token_position != previous_position + 1) {
                throw std::runtime_error("captured observation positions are not contiguous");
            }
            if (event.chunk_index < previous_chunk) {
                throw std::runtime_error("captured observation chunk order is invalid");
            }
            previous_position = event.token_position;
            previous_chunk = event.chunk_index;

            if (event.observed_token < 0 || event.baseline_greedy < 0 || !model) {
                throw std::runtime_error("captured observation contains an invalid token");
            }
            if (event.query.size() != size_t(state->width)) {
                throw std::runtime_error("captured observation query width mismatch");
            }

            double raw_norm = 0.0;
            double budget = 0.0;
            std::string source;
            state->retrieve(event.query.data(), raw_norm, budget, source);
            ++state->retrievals;
            ++state->query_captures;
            state->sum_b += budget;
            state->max_b = std::max(state->max_b, budget);

            std::vector<float> observed_row(size_t(state->width));
            std::vector<float> baseline_row(size_t(state->width));
            if (!llama_model_output_row(model, event.observed_token,
                                        observed_row.data(), observed_row.size()) ||
                    !llama_model_output_row(model, event.baseline_greedy,
                                            baseline_row.data(), baseline_row.size())) {
                throw std::runtime_error(
                    "Step-30 output projection is unavailable for a captured token");
            }

            std::vector<float> direction(observed_row.size());
            for (size_t i = 0; i < direction.size(); ++i) {
                direction[i] = observed_row[i] - baseline_row[i];
            }
            const double direction_norm = vector_norm(direction);
            const auto counts_it = state->writer_counts.find(source);
            const step30_counts counts_before = counts_it == state->writer_counts.end() ?
                step30_counts{} : counts_it->second;
            const step30_gate gate = compute_step30_gate(counts_before);
            const std::string event_hash_before = per_event_hash ?
                state->state_hash() : "deferred_prototype_hash";

            auto & counts = state->writer_counts[source];
            ++counts.total;
            ++counts.matching;

            const bool prediction_match = event.observed_token == event.baseline_greedy;
            prediction_matches += prediction_match ? 1 : 0;
            gate_sum += gate.gate;

            bool updated = false;
            double coefficient = 0.0;
            double update_norm = 0.0;
            std::vector<float> normalized_direction;
            sidecar::prototype_update prototype_result;
            if (std::isfinite(direction_norm) && direction_norm > EPSILON) {
                sidecar::component component;
                component.coefficient = ETA * gate.gate;
                component.source = source;
                component.destination = event.observed_token_text.empty() ?
                    "token:" + std::to_string(event.observed_token) : event.observed_token_text;
                component.key = event.query;
                component.value = std::move(direction);
                for (float & value : component.value) {
                    value = float(double(value) / direction_norm);
                }
                if (capture_debug_vectors) {
                    normalized_direction = component.value;
                }
                coefficient = component.coefficient;
                update_norm = coefficient * vector_norm(component.key);
                updated = coefficient > 0.0;
                if (updated) {
                    if (state->is_prototype_backend()) {
                        prototype_result = state->absorb_prototype(
                            component.key, component.value, component.coefficient);
                    } else {
                        state->append_component(std::move(component));
                    }
                    ++state->writes;
                    ++writes_realized;
                    update_norm_sum += update_norm;
                }
            }

            json event_result = json{
                {"episode_id", episode_id},
                {"token_position", event.token_position},
                {"chunk_index", event.chunk_index},
                {"observed_token_id", event.observed_token},
                {"observed_token", event.observed_token_text},
                {"baseline_greedy_token_id", event.baseline_greedy},
                {"baseline_greedy_token", event.baseline_token_text},
                {"prediction_match", prediction_match},
                {"updated", updated},
                {"G", gate.gate},
                {"novelty", gate.novelty},
                {"surprise", gate.surprise},
                {"recurrence", gate.recurrence},
                {"consistency", gate.consistency},
                {"effective_eta", coefficient},
                {"write_direction_norm", direction_norm},
                {"update_norm", update_norm},
                {"source", source},
                {"sha256_before", event_hash_before},
                {"sha256_after", per_event_hash ? state->state_hash() : "deferred_prototype_hash"},
                {"components", state->components.size()},
                {"allocated_prototypes", state->prototypes.size()},
                {"active_prototypes", state->active_prototype_count()},
                {"prototype_index", updated && state->is_prototype_backend() ?
                    static_cast<int64_t>(prototype_result.index) : -1},
                {"prototype_similarity", updated && state->is_prototype_backend() ?
                    prototype_result.similarity : 0.0},
                {"prototype_created", updated && state->is_prototype_backend() &&
                    prototype_result.created},
                {"forced_merge", updated && state->is_prototype_backend() &&
                    prototype_result.forced_merge},
                {"count_total", counts.total},
                {"count_matching", counts.matching},
            };
            if (capture_debug_vectors) {
                event_result["debug_key"] = event.query;
                event_result["debug_write_direction"] = std::move(normalized_direction);
            }
            event_results.push_back(std::move(event_result));
        }
    } catch (...) {
        state->components.resize(components_before);
        state->writer_counts = writer_counts_before;
        state->state_hash_context = hash_context_before;
        state->writes = writes_before;
        if (state->is_prototype_backend()) {
            state->prototypes = prototypes_before;
            state->lifetime_merges = lifetime_merges_before;
            state->forced_merges = forced_merges_before;
            state->total_created_prototypes = created_prototypes_before;
            state->experiences_absorbed = experiences_absorbed_before;
            state->prototype_hash_dirty = prototype_hash_dirty_before;
            state->prototype_hash_cache = prototype_hash_cache_before;
        }
        state->retrievals = retrievals_before;
        state->sum_b = sum_b_before;
        state->max_b = max_b_before;
        state->query_captures = query_captures_before;
        throw;
    }

    const std::string state_after = state->state_hash();
    return json{
        {"id", id},
        {"episode_id", episode_id},
        {"learning_mode", learning_mode},
        {"chunk_size", chunk_size},
        {"tokens_observed", events.size()},
        {"write_opportunities", events.size()},
        {"writes_realized", writes_realized},
        {"prediction_matches", prediction_matches},
        {"components_added", state->components.size() - components_before},
        {"components_before", components_before},
        {"components_after", state->components.size()},
        {"memory_backend", state->memory_backend},
        {"allocated_prototypes", state->prototypes.size()},
        {"active_prototypes", state->active_prototype_count()},
        {"prototypes_created", state->total_created_prototypes - created_prototypes_before},
        {"prototype_merges", state->lifetime_merges - lifetime_merges_before},
        {"forced_merges", state->forced_merges - forced_merges_before},
        {"mean_gate", gate_sum / double(events.size())},
        {"mean_update_norm", writes_realized == 0 ? 0.0 : update_norm_sum / double(writes_realized)},
        {"updated", writes_realized > 0},
        {"sha256_before", state_before},
        {"sha256_after", state_after},
        {"events", std::move(event_results)},
    };
}

json server_sidecar_manager::observe_sham(
        const std::string & id,
        const std::string & episode_id,
        llama_token observed_token,
        const std::string & observed_target,
        const std::vector<float> & write_direction,
        double gate,
        double effective_eta) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (!state->learning_enabled) {
        return json{{"id", id}, {"episode_id", episode_id}, {"updated", false}, {"reason", "learning_disabled"}};
    }
    auto episode_it = state->episodes.find(episode_id);
    if (episode_it == state->episodes.end()) {
        throw std::runtime_error("invalid or expired episode_id");
    }
    if (episode_it->second.version != state->version) {
        throw std::runtime_error("episode invalidated by sidecar state change");
    }
    if (observed_token < 0 || write_direction.size() != size_t(state->width) ||
            !std::isfinite(gate) || !std::isfinite(effective_eta) || effective_eta < 0.0) {
        throw std::runtime_error("invalid SHAM write direction or dose");
    }

    std::vector<float> direction = write_direction;
    const double direction_norm = vector_norm(direction);
    const std::string source_key = episode_it->second.source;
    const llama_token baseline_greedy = episode_it->second.baseline_greedy;
    std::vector<float> write_key = episode_it->second.query;
    const std::string before = state->state_hash();

    state->episodes.erase(episode_it);
    auto & counts = state->writer_counts[source_key];
    ++counts.total;
    ++counts.matching;

    if (!std::isfinite(direction_norm) || direction_norm <= EPSILON) {
        return json{
            {"id", id},
            {"episode_id", episode_id},
            {"updated", false},
            {"reason", "zero_direction"},
            {"observed_token_id", observed_token},
            {"baseline_greedy_token_id", baseline_greedy},
            {"G", gate},
            {"effective_eta", effective_eta},
            {"write_direction_norm", direction_norm},
            {"sha256_before", before},
            {"sha256_after", state->state_hash()},
            {"components", state->components.size()},
            {"count_total", counts.total},
            {"count_matching", counts.matching},
        };
    }

    sidecar::component component;
    component.coefficient = effective_eta;
    component.source = source_key;
    component.destination = observed_target.empty() ?
        "token:" + std::to_string(observed_token) : observed_target;
    component.key = std::move(write_key);
    component.value = std::move(direction);
    for (float & value : component.value) {
        value = float(double(value) / direction_norm);
    }
    const double update_norm = component.coefficient * vector_norm(component.key);
    const bool updated = component.coefficient > 0.0;
    if (updated) {
        state->append_component(std::move(component));
        ++state->writes;
    }
    const std::string after = state->state_hash();
    return json{
        {"id", id},
        {"episode_id", episode_id},
        {"updated", updated},
        {"observed_token_id", observed_token},
        {"baseline_greedy_token_id", baseline_greedy},
        {"G", gate},
        {"effective_eta", effective_eta},
        {"write_direction_norm", direction_norm},
        {"update_norm", update_norm},
        {"sha256_before", before},
        {"sha256_after", after},
        {"components", state->components.size()},
        {"count_total", counts.total},
        {"count_matching", counts.matching},
        {"mode", "sham_write_direction"},
    };
}

bool server_sidecar_manager::discard_episode(const std::string & id, const std::string & episode_id) {
    if (episode_id.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex);
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    return state->episodes.erase(episode_id) != 0;
}

json server_sidecar_manager::observe_debug(
        const std::string & id,
        const std::string & episode_id,
        const std::string & observed_target,
        const std::string & source,
        const std::vector<float> & write_direction,
        double novelty,
        double surprise,
        double recurrence,
        double consistency) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (!state->learning_enabled) {
        return json{{"id", id}, {"episode_id", episode_id}, {"updated", false}, {"reason", "learning_disabled"}};
    }
    auto episode_it = state->episodes.find(episode_id);
    if (episode_it == state->episodes.end()) {
        throw std::runtime_error("invalid or expired episode_id");
    }
    if (episode_it->second.version != state->version) {
        throw std::runtime_error("episode invalidated by sidecar state change");
    }
    if (observed_target.empty()) {
        throw std::runtime_error("observe requires observed_target");
    }
    if (write_direction.size() != size_t(state->width)) {
        throw std::runtime_error("debug_write_direction requires model-width vector");
    }
    const double gate = std::clamp(0.25 * std::max(novelty, surprise) + recurrence * consistency, 0.0, 1.0);
    const double direction_norm = vector_norm(write_direction);
    if (!std::isfinite(direction_norm) || direction_norm <= EPSILON) {
        throw std::runtime_error("debug_write_direction has zero or non-finite norm");
    }
    sidecar::component component;
    component.coefficient = ETA * gate;
    component.source = source.empty() ? episode_it->second.source : source;
    component.destination = observed_target;
    component.key = episode_it->second.query;
    component.value = write_direction;
    for (float & value : component.value) {
        value = float(double(value) / direction_norm);
    }
    const std::string before = state->state_hash();
    const double update_norm = ETA * gate * vector_norm(component.key);
    const double coefficient = component.coefficient;
    const bool updated = coefficient > 0.0;
    if (updated) {
        state->append_component(std::move(component));
    }
    state->episodes.erase(episode_it);
    if (updated) {
        ++state->writes;
    }
    const std::string after = state->state_hash();
    return json{
        {"id", id},
        {"episode_id", episode_id},
        {"updated", updated},
        {"mode", "debug_write_vector"},
        {"G", gate},
        {"effective_eta", ETA * gate},
        {"write_direction_norm", direction_norm},
        {"update_norm", update_norm},
        {"sha256_before", before},
        {"sha256_after", after},
        {"components", state->components.size()},
    };
}

bool server_sidecar_manager::apply_query(
        const std::string & id,
        llama_context * ctx,
        const float * query,
        int32_t width,
        std::string * episode_id,
        std::string & error) {
    if (ctx == nullptr || query == nullptr) {
        error = "sidecar query context is unavailable";
        return false;
    }
    server_sidecar_runtime_step step;
    if (!prepare_runtime_step(id, query, width, episode_id, -1, step, error)) {
        return false;
    }
    if (step.raw_norm <= EPSILON) {
        return server_sidecar_injection_engine::clear_internal_cvec(ctx, width, error);
    }

    std::lock_guard<std::mutex> lock(mutex);
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    return server_sidecar_injection_engine::apply_internal_cvec(
        ctx, *injection_profile, step.direction.data(), width, model_layers, step.budget,
        state->dispatch_telemetry, error);
}

bool server_sidecar_manager::prepare_runtime_step(
        const std::string & id,
        const float * query,
        int32_t width,
        std::string * episode_id,
        llama_token baseline_greedy,
        server_sidecar_runtime_step & step,
        std::string & error,
        int32_t forced_prototype_id,
        bool force_native_current,
        const std::vector<float> * forced_direction) {
    if (query == nullptr) {
        error = "sidecar query is unavailable";
        return false;
    }
    if (forced_direction != nullptr &&
            (forced_prototype_id >= 0 || force_native_current)) {
        error = "direct-vector forcing cannot be combined with prototype forcing";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (injection_profile == nullptr) {
        error = profile_error.empty() ?
            "sidecar_injection_profile_not_supported" : profile_error;
        return false;
    }
    if (model_width != width) {
        error = "sidecar query width does not match model width";
        return false;
    }
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    double raw_norm = 0.0;
    double budget = 0.0;
    std::string source;
    std::vector<float> response;
    if (forced_direction != nullptr) {
        if (forced_direction->size() != size_t(width)) {
            error = "direct-vector forcing width does not match model width";
            return false;
        }
        response = *forced_direction;
        for (const float value : response) {
            if (!std::isfinite(value)) {
                error = "direct-vector forcing contains a non-finite value";
                return false;
            }
        }
        raw_norm = vector_norm(response);
        budget = server_sidecar_authority_controller::budget(raw_norm);
        source = "direct_vector";
    } else {
        const std::vector<float> normal_response = state->retrieve(query, raw_norm, budget, source);
        response = normal_response;
    }
    if (forced_direction == nullptr && !force_native_current && forced_prototype_id >= 0) {
        double forced_raw_norm = 0.0;
        double forced_budget = 0.0;
        std::string forced_source;
        response = state->retrieve_forced(
            query,
            size_t(forced_prototype_id),
            raw_norm,
            forced_raw_norm,
            forced_budget,
            forced_source,
            error);
        if (response.empty()) {
            if (error.empty()) {
                error = "forced prototype retrieval returned no direction";
            }
            return false;
        }
        raw_norm = forced_raw_norm;
        budget = forced_budget;
        source = std::move(forced_source);
    }
    ++state->retrievals;
    state->sum_b += budget;
    state->max_b = std::max(state->max_b, budget);

    if (episode_id != nullptr) {
        if (episode_id->empty()) {
            *episode_id = id + "-episode-" + std::to_string(state->next_episode++);
        }
        // A debug-only frozen capture still needs a request-local episode so
        // observe() can return the query.  observe() removes this record
        // without mutating sidecar learning state when learning is disabled.
        state->episodes[*episode_id] = sidecar::episode{
            std::vector<float>(query, query + width),
            source, baseline_greedy, state->version,
        };
    }

    step.raw_norm = raw_norm;
    step.budget = budget;
    step.source = std::move(source);
    if (raw_norm <= EPSILON) {
        step.direction.clear();
        return true;
    }
    const double inverse_norm = 1.0 / raw_norm;
    step.direction.resize(response.size());
    for (size_t i = 0; i < response.size(); ++i) {
        step.direction[i] = float(double(response[i]) * inverse_norm);
    }
    return true;
}

bool server_sidecar_manager::apply_runtime_hidden(
        const std::string & id,
        llama_context * ctx,
        const float * direction,
        int32_t width,
        const std::array<double, 2> & alphas,
        std::string & error) {
    if (ctx == nullptr || direction == nullptr) {
        error = "sidecar injection context or direction is unavailable";
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (injection_profile == nullptr) {
        error = profile_error.empty() ?
            "sidecar_injection_profile_not_supported" : profile_error;
        return false;
    }
    if (model_width != width) {
        error = "sidecar injection width does not match model width";
        return false;
    }
    const auto state = get(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    return server_sidecar_injection_engine::apply_internal_cvec_alphas(
        ctx, *injection_profile, direction, width, model_layers, alphas,
        state->dispatch_telemetry, error);
}

bool server_sidecar_manager::clear_runtime_hidden(
        llama_context * ctx,
        int32_t width,
        std::string & error) const {
    return server_sidecar_injection_engine::clear_internal_cvec(ctx, width, error);
}

bool server_sidecar_manager::apply_runtime_step30(
        const std::string & id,
        llama_context * ctx,
        const float * direction,
        int32_t width,
        const std::array<double, 3> & rho,
        std::string & error) {
    if (ctx == nullptr || direction == nullptr) {
        error = "sidecar Step-30 context or direction is unavailable";
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (injection_profile == nullptr) {
        error = profile_error.empty() ?
            "sidecar_injection_profile_not_supported" : profile_error;
        return false;
    }
    if (model_width != width) {
        error = "sidecar Step-30 width does not match model width";
        return false;
    }
    const auto it = sidecars.find(id);
    if (it == sidecars.end()) {
        error = "unknown sidecar_id: " + id;
        return false;
    }

    if (llama_set_step30_intervention(
                ctx,
                direction,
                size_t(width),
                float(rho[0]),
                float(rho[1]),
                float(rho[2])) != 0) {
        error = "failed to configure graph-native Step-30 intervention";
        return false;
    }
    std::lock_guard<std::mutex> state_lock(it->second->mutex);
    it->second->dispatch_telemetry.injection_sites += 2;
    it->second->dispatch_telemetry.graph_layer_24 += 1;
    it->second->dispatch_telemetry.graph_layer_28 += 1;
    return true;
}

void server_sidecar_manager::clear_runtime_step30(llama_context * ctx) const {
    if (ctx != nullptr) {
        llama_clear_step30_intervention(ctx);
    }
}

bool server_sidecar_manager::validate_request(
        const std::string & id,
        int32_t n_parallel,
        int32_t n_completions,
        int32_t n_predict,
        bool speculative,
        bool capture_episode,
        bool internal_plasticity,
        bool observed_sequence,
        std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!model || model_width <= 0) {
        error = "sidecar manager is not bound to a model";
        return false;
    }
    if (root.empty()) {
        error = "sidecar integration is disabled; pass --sidecar-dir";
        return false;
    }
    if (injection_profile == nullptr) {
        error = profile_error.empty() ?
            "sidecar_injection_profile_not_supported" : profile_error;
        return false;
    }
    if (sidecars.find(id) == sidecars.end()) {
        error = "unknown sidecar_id: " + id;
        return false;
    }
    if (n_parallel != 1) {
        error = "sidecar requests require --parallel 1 because the functional "
                "Step-30 runtime owns the full slot transaction";
        return false;
    }
    if (n_completions != 1) {
        error = "sidecar requests require n=1 / one completion";
        return false;
    }
    const auto state = sidecars.at(id);
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (capture_episode && !state->learning_enabled) {
        error = "sidecar_capture_episode requires persistent learning to be enabled";
        return false;
    }
    if (capture_episode && (internal_plasticity || observed_sequence)) {
        error = "sidecar_capture_episode cannot be combined with another learning mode";
        return false;
    }
    if (internal_plasticity && n_predict <= 0) {
        error = "internal reasoning plasticity requires a finite positive max_tokens limit";
        return false;
    }
    if (internal_plasticity && observed_sequence) {
        error = "sidecar observed-token sequences cannot enable internal reasoning plasticity";
        return false;
    }
    if (capture_episode && n_predict != 1) {
        error = "sidecar_capture_episode requires exactly one generated token per request "
                "(set max_tokens=1)";
        return false;
    }
    if (speculative) {
        error = "sidecar requests do not support speculative decoding in Phase 1";
        return false;
    }
    return true;
}
