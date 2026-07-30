#include "llama-sparse-residual.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr std::array<char, 8> sparse_magic = {'S', 'P', 'R', 'S', 'R', 'E', 'S', '1'};
constexpr uint32_t sparse_version = 1;

struct sparse_record {
    std::string name;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;
    uint64_t nnz = 0;
    std::vector<uint32_t> row_ptr;
    std::vector<uint16_t> col_idx;
    std::vector<uint16_t> values;
    const ggml_tensor * weight = nullptr;
};

class binary_reader {
public:
    explicit binary_reader(std::vector<uint8_t> bytes) : bytes(std::move(bytes)) {}

    template<typename T>
    T read() {
        static_assert(std::is_trivially_copyable<T>::value, "binary scalar required");
        require(sizeof(T));
        T value;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }

    template<typename T>
    std::vector<T> read_vector(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("sparse residual vector size overflow");
        }
        const size_t n_bytes = count * sizeof(T);
        require(n_bytes);
        std::vector<T> result(count);
        if (n_bytes > 0) {
            std::memcpy(result.data(), bytes.data() + offset, n_bytes);
        }
        offset += n_bytes;
        return result;
    }

    std::string read_string(size_t length) {
        require(length);
        std::string result(reinterpret_cast<const char *>(bytes.data() + offset), length);
        offset += length;
        return result;
    }

    size_t remaining() const {
        return bytes.size() - offset;
    }

private:
    void require(size_t count) const {
        if (count > bytes.size() - offset) {
            throw std::runtime_error("truncated sparse residual sidecar");
        }
    }

    std::vector<uint8_t> bytes;
    size_t offset = 0;
};

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open sparse residual sidecar: " + path);
    }

    const std::streamsize size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot determine sparse residual sidecar size: " + path);
    }
    input.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0 && !input.read(reinterpret_cast<char *>(bytes.data()), size)) {
        throw std::runtime_error("cannot read sparse residual sidecar: " + path);
    }
    return bytes;
}

std::string hex_string(const std::vector<uint8_t> & bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (uint8_t value : bytes) {
        result.push_back(digits[value >> 4]);
        result.push_back(digits[value & 0x0f]);
    }
    return result;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

ggml_backend_buffer_type_t tensor_buft(const ggml_tensor * tensor) {
    const ggml_tensor * storage = tensor;
    while (storage != nullptr && storage->buffer == nullptr && storage->view_src != nullptr) {
        storage = storage->view_src;
    }
    if (storage == nullptr || storage->buffer == nullptr) {
        throw std::runtime_error("model tensor has no backend buffer: " + std::string(tensor->name));
    }
    return ggml_backend_buffer_get_type(storage->buffer);
}

} // namespace

struct llama_sparse_residual::impl {
    std::vector<ggml_context_ptr> contexts;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::unordered_map<const ggml_tensor *, llama_sparse_residual_entry> entries;
};

llama_sparse_residual::llama_sparse_residual(std::unique_ptr<impl> pimpl) :
    pimpl(std::move(pimpl)) {
}

llama_sparse_residual::~llama_sparse_residual() = default;

std::unique_ptr<llama_sparse_residual> llama_sparse_residual::load(
        const std::string & path,
        const llama_model & model) {
    binary_reader reader(read_file(path));

    const auto magic = reader.read_vector<uint8_t>(sparse_magic.size());
    if (!std::equal(magic.begin(), magic.end(), sparse_magic.begin())) {
        throw std::runtime_error("invalid sparse residual magic");
    }

    const uint32_t version = reader.read<uint32_t>();
    const uint32_t tensor_count = reader.read<uint32_t>();
    const uint64_t declared_nnz = reader.read<uint64_t>();
    const auto source_gguf_sha = reader.read_vector<uint8_t>(32);
    const auto source_npz_sha = reader.read_vector<uint8_t>(32);
    const uint32_t metadata_length = reader.read<uint32_t>();

    if (version != sparse_version) {
        throw std::runtime_error("unsupported sparse residual version: " + std::to_string(version));
    }
    if (tensor_count == 0) {
        throw std::runtime_error("sparse residual contains no tensors");
    }
    reader.read_string(metadata_length);

    const std::string source_gguf_hex = hex_string(source_gguf_sha);
    if (const char * expected = std::getenv("LLAMA_SPARSE_RESIDUAL_MODEL_SHA256")) {
        if (expected[0] != '\0' && lowercase(expected) != source_gguf_hex) {
            throw std::runtime_error("sparse residual source GGUF SHA-256 mismatch");
        }
    }

    std::vector<sparse_record> records;
    records.reserve(tensor_count);
    std::unordered_set<const ggml_tensor *> seen_weights;
    uint64_t observed_nnz = 0;

    for (uint32_t i = 0; i < tensor_count; ++i) {
        const uint16_t name_length = reader.read<uint16_t>();
        sparse_record record;
        record.n_rows = reader.read<uint32_t>();
        record.n_cols = reader.read<uint32_t>();
        record.nnz = reader.read<uint64_t>();
        const uint8_t col_type = reader.read<uint8_t>();
        const uint8_t value_type = reader.read<uint8_t>();

        if (name_length == 0 || record.n_rows == 0 || record.n_cols == 0) {
            throw std::runtime_error("invalid sparse residual tensor dimensions");
        }
        if (col_type != 2 || value_type != 1) {
            throw std::runtime_error("unsupported sparse residual index/value encoding");
        }
        if (record.nnz > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("sparse residual tensor exceeds uint32 CSR capacity");
        }

        record.name = reader.read_string(name_length);
        record.row_ptr = reader.read_vector<uint32_t>(static_cast<size_t>(record.n_rows) + 1);
        record.col_idx = reader.read_vector<uint16_t>(static_cast<size_t>(record.nnz));
        record.values = reader.read_vector<uint16_t>(static_cast<size_t>(record.nnz));

        if (record.row_ptr.front() != 0 || record.row_ptr.back() != record.nnz ||
                !std::is_sorted(record.row_ptr.begin(), record.row_ptr.end())) {
            throw std::runtime_error("invalid CSR row pointers for tensor: " + record.name);
        }
        if (std::any_of(record.col_idx.begin(), record.col_idx.end(), [&](uint16_t col) {
                    return col >= record.n_cols;
                })) {
            throw std::runtime_error("CSR column out of bounds for tensor: " + record.name);
        }

        record.weight = model.get_tensor(record.name.c_str());
        if (record.weight == nullptr) {
            throw std::runtime_error("sparse residual tensor absent from model: " + record.name);
        }
        if (record.weight->ne[0] != record.n_cols || record.weight->ne[1] != record.n_rows ||
                record.weight->ne[2] != 1 || record.weight->ne[3] != 1) {
            throw std::runtime_error("sparse residual shape mismatch for tensor: " + record.name);
        }
        if (!seen_weights.insert(record.weight).second) {
            throw std::runtime_error("duplicate sparse residual tensor: " + record.name);
        }

        if (observed_nnz > declared_nnz || record.nnz > declared_nnz - observed_nnz) {
            throw std::runtime_error("sparse residual NNZ overflow");
        }
        observed_nnz += record.nnz;
        records.push_back(std::move(record));
    }

    if (reader.remaining() != 0) {
        throw std::runtime_error("sparse residual contains trailing bytes");
    }
    if (observed_nnz != declared_nnz) {
        throw std::runtime_error("sparse residual total NNZ mismatch");
    }

    auto result = std::make_unique<impl>();
    std::map<ggml_backend_buffer_type_t, std::vector<size_t>> groups;
    for (size_t i = 0; i < records.size(); ++i) {
        groups[tensor_buft(records[i].weight)].push_back(i);
    }

    for (const auto & [buft, indices] : groups) {
        const size_t tensor_count_for_buft = indices.size() * 3;
        ggml_init_params params = {
            tensor_count_for_buft * ggml_tensor_overhead() + 4096,
            nullptr,
            true,
        };
        ggml_context_ptr context { ggml_init(params) };
        if (!context) {
            throw std::runtime_error("failed to initialize sparse residual GGML context");
        }

        for (size_t index : indices) {
            const auto & record = records[index];
            auto & entry = result->entries[record.weight];
            entry.n_rows = record.n_rows;
            entry.n_cols = record.n_cols;
            entry.row_ptr = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, record.row_ptr.size());
            entry.col_idx = ggml_new_tensor_1d(context.get(), GGML_TYPE_I16, record.col_idx.size());
            entry.values = ggml_new_tensor_1d(context.get(), GGML_TYPE_F16, record.values.size());
            ggml_set_name(entry.row_ptr, (record.name + ".sparse.row_ptr").c_str());
            ggml_set_name(entry.col_idx, (record.name + ".sparse.col_idx").c_str());
            ggml_set_name(entry.values,  (record.name + ".sparse.values").c_str());
        }

        ggml_backend_buffer_ptr buffer { ggml_backend_alloc_ctx_tensors_from_buft(context.get(), buft) };
        if (!buffer) {
            throw std::runtime_error("failed to allocate sparse residual backend buffer");
        }

        for (size_t index : indices) {
            const auto & record = records[index];
            const auto & entry = result->entries.at(record.weight);
            ggml_backend_tensor_set(entry.row_ptr, record.row_ptr.data(), 0, record.row_ptr.size() * sizeof(uint32_t));
            ggml_backend_tensor_set(entry.col_idx, record.col_idx.data(), 0, record.col_idx.size() * sizeof(uint16_t));
            ggml_backend_tensor_set(entry.values, record.values.data(), 0, record.values.size() * sizeof(uint16_t));
        }

        result->contexts.push_back(std::move(context));
        result->buffers.push_back(std::move(buffer));
    }

    LLAMA_LOG_INFO(
            "%s: loaded %u tensors, %" PRIu64 " residual weights (GGUF SHA-256 %s, NPZ SHA-256 %s)\n",
            __func__,
            tensor_count,
            observed_nnz,
            source_gguf_hex.c_str(),
            hex_string(source_npz_sha).c_str());
    return std::unique_ptr<llama_sparse_residual>(new llama_sparse_residual(std::move(result)));
}

const llama_sparse_residual_entry * llama_sparse_residual::get(const ggml_tensor * weight) const {
    const auto it = pimpl->entries.find(weight);
    return it == pimpl->entries.end() ? nullptr : &it->second;
}
