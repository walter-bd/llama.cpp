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
    std::vector<component> components;
    std::unordered_map<std::string, episode> episodes;
    std::unordered_map<std::string, step30_counts> writer_counts;
    uint64_t next_episode = 1;
    uint64_t version = 0;
    uint64_t retrievals = 0;
    uint64_t writes = 0;
    double sum_b = 0.0;
    double max_b = 0.0;
    std::string loaded_path;
    std::string package_sha;
    std::string base_model_identifier = "unavailable";
    std::string base_model_sha256;
    std::string injection_profile_id;
    uint64_t query_captures = 0;
    server_sidecar_dispatch_telemetry dispatch_telemetry;

    std::vector<float> retrieve(
            const float * query,
            double & raw_norm,
            double & budget,
            std::string & source) const {
        std::vector<float> response(width, 0.0f);
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

    std::string state_hash() const {
        std::vector<uint8_t> bytes;
        for (const component & component : components) {
            const std::string metadata = component.source + "\n" + component.destination + "\n" +
                std::to_string(component.coefficient) + "\n";
            bytes.insert(bytes.end(), metadata.begin(), metadata.end());
            const auto * value_begin = reinterpret_cast<const uint8_t *>(component.value.data());
            bytes.insert(bytes.end(), value_begin, value_begin + component.value.size() * sizeof(float));
            const auto * key_begin = reinterpret_cast<const uint8_t *>(component.key.data());
            bytes.insert(bytes.end(), key_begin, key_begin + component.key.size() * sizeof(float));
        }
        return sha256_bytes(bytes);
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
                {"format_version", 1},
                {"learning_enabled", state.learning_enabled},
                {"sha256", state.state_hash()},
                {"components", state.components.size()},
                {"width", state.width},
                {"bytes", state.components.size() * size_t(state.width) * sizeof(float) * 2},
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
        {"format_version", 1},
        {"learning_enabled", state->learning_enabled},
        {"sha256", state->state_hash()},
        {"components", state->components.size()},
        {"width", state->width},
        {"bytes", state->components.size() * size_t(state->width) * sizeof(float) * 2},
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

json server_sidecar_manager::create(const std::string & id, bool learning_enabled) {
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
    auto state = std::make_shared<sidecar>();
    state->id = id;
    state->width = model_width;
    state->learning_enabled = learning_enabled;
    state->injection_profile_id = injection_profile->id;
    sidecars.emplace(id, state);
    return json{{"id", id}, {"loaded", true}, {"learning_enabled", learning_enabled}, {"components", 0}, {"width", model_width}};
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
    const fs::path components_path = package / "components.jsonl";
    const fs::path vectors_path = package / "vectors.f32";

    const auto manifest_bytes = read_bytes(manifest_path);
    const auto components_bytes = read_bytes(components_path);
    const auto vectors_bytes = read_bytes(vectors_path);
    const json manifest = json::parse(std::string(manifest_bytes.begin(), manifest_bytes.end()));
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
        state->components.push_back(std::move(component));
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
    state->components.clear();
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
        const std::string & observed_target) {
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
    const double update_norm = component.coefficient * vector_norm(component.key);
    const double coefficient = component.coefficient;
    const bool updated = coefficient > 0.0;
    if (updated) {
        state->components.push_back(std::move(component));
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
        {"components", state->components.size()},
        {"count_total", counts.total},
        {"count_matching", counts.matching},
    };
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
        state->components.push_back(std::move(component));
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
        std::string & error) {
    if (query == nullptr) {
        error = "sidecar query is unavailable";
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
    const std::vector<float> response = state->retrieve(query, raw_norm, budget, source);
    ++state->retrievals;
    state->sum_b += budget;
    state->max_b = std::max(state->max_b, budget);

    if (episode_id != nullptr) {
        if (!state->learning_enabled) {
            episode_id->clear();
        } else {
            if (episode_id->empty()) {
                *episode_id = id + "-episode-" + std::to_string(state->next_episode++);
            }
            state->episodes[*episode_id] = sidecar::episode{
                std::vector<float>(query, query + width),
                source, baseline_greedy, state->version,
            };
        }
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

bool server_sidecar_manager::validate_request(
        const std::string & id,
        int32_t n_parallel,
        int32_t n_completions,
        int32_t n_predict,
        bool speculative,
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
    if (state->learning_enabled && n_predict != 1) {
        error = "experience-derived sidecar learning requires exactly one generated "
                "token per request (set max_tokens=1); disable learning for "
                "multi-token inference";
        return false;
    }
    if (speculative) {
        error = "sidecar requests do not support speculative decoding in Phase 1";
        return false;
    }
    return true;
}
