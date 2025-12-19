#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <thread>
#include <future>
#include <zstd.h>

/**
 * Optimized Advanced LLM Codec for SafeTensors compression
 * 
 * Optimizations:
 * 1. ZSTD compression with level 4 (faster than DEFLATE, better ratio)
 * 2. Parallel block compression using multiple threads
 * 3. Optimized float16 conversion
 * 4. Better memory management
 */

class OptimizedLLMCodec {
private:
    struct Header {
        uint64_t original_size;
        uint64_t json_header_size;
        uint32_t num_floats;
        uint32_t num_blocks;
        uint64_t compressed_tensor_size;
    };

    struct BlockHeader {
        uint64_t compressed_size;
        uint64_t original_size;
    };

        static void xor_delta_encode_inplace_u32(std::vector<uint32_t>& data) {
        if (data.size() <= 1) return;
        for (size_t i = data.size() - 1; i > 0; i--) {
            data[i] ^= data[i - 1];
        }
    }

    static void xor_delta_decode_inplace_u32(std::vector<uint32_t>& data) {
        if (data.size() <= 1) return;
        for (size_t i = 1; i < data.size(); i++) {
            data[i] ^= data[i - 1];
        }
    }

    // ZSTD single-frame, multi-thread compression for better ratio
    static std::vector<uint8_t> zstd_compress_mt(const uint8_t* data, size_t size, int level, int workers, bool enableLDM) {
        std::vector<uint8_t> out(ZSTD_compressBound(size));
        ZSTD_CCtx* cctx = ZSTD_createCCtx();
        if (!cctx) return {};

        ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, workers > 0 ? workers : 0);
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_enableLongDistanceMatching, enableLDM ? 1 : 0);
        // Optionally increase window for large tensors
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_windowLog, 27); // ~128MB window

        size_t res = ZSTD_compress2(cctx, out.data(), out.size(), data, size);
        if (ZSTD_isError(res)) {
            std::cerr << "ZSTD compress2 failed: " << ZSTD_getErrorName(res) << std::endl;
            ZSTD_freeCCtx(cctx);
            return {};
        }
        out.resize(res);
        ZSTD_freeCCtx(cctx);
        return out;
    }

    static std::vector<uint8_t> zstd_decompress(const uint8_t* data, size_t compressed_size, size_t original_size) {
        std::vector<uint8_t> out(original_size);
        size_t res = ZSTD_decompress(out.data(), original_size, data, compressed_size);
        if (ZSTD_isError(res)) {
            std::cerr << "ZSTD decompress failed: " << ZSTD_getErrorName(res) << std::endl;
            return {};
        }
        return out;
    }

    // Optimized float32 to float16 (branchless where possible)
    static uint16_t float32_to_float16(float value) {
        uint32_t f32;
        std::memcpy(&f32, &value, sizeof(float));
        
        uint32_t sign = (f32 >> 16) & 0x8000;
        int32_t exp = ((f32 >> 23) & 0xff) - 127;
        uint32_t mantissa = f32 & 0x7fffff;
        
        if (exp <= -15) return sign;
        if (exp >= 16) return sign | 0x7c00;
        
        exp += 15;
        mantissa >>= 13;
        
        return sign | (exp << 10) | mantissa;
    }

    static float float16_to_float32(uint16_t f16) {
        uint32_t sign = (f16 & 0x8000) << 16;
        int32_t exp = (f16 >> 10) & 0x1f;
        uint32_t mantissa = f16 & 0x3ff;
        
        if (exp == 0) {
            if (mantissa == 0) {
                uint32_t f32 = sign;
                float result;
                std::memcpy(&result, &f32, sizeof(float));
                return result;
            }
            return 0.0f;
        } else if (exp == 31) {
            uint32_t f32 = sign | 0x7f800000 | (mantissa << 13);
            float result;
            std::memcpy(&result, &f32, sizeof(float));
            return result;
        }
        
        exp = exp - 15 + 127;
        uint32_t f32 = sign | (exp << 23) | (mantissa << 13);
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    }

    // Delta encoding
    static void delta_encode_inplace(std::vector<uint16_t>& data) {
        if (data.size() <= 1) return;
        
        for (size_t i = data.size() - 1; i > 0; i--) {
            int32_t delta = static_cast<int32_t>(data[i]) - static_cast<int32_t>(data[i-1]);
            data[i] = static_cast<uint16_t>(delta);
        }
    }

    // Delta decoding
    static void delta_decode_inplace(std::vector<uint16_t>& data) {
        if (data.size() <= 1) return;
        
        for (size_t i = 1; i < data.size(); i++) {
            int32_t value = static_cast<int32_t>(data[i-1]) + static_cast<int16_t>(data[i]);
            data[i] = static_cast<uint16_t>(value);
        }
    }

    // Compress a single block with ZSTD level 4
    static std::vector<uint8_t> compress_block(const uint8_t* data, size_t size) {
        size_t compressed_size = ZSTD_compressBound(size);
        std::vector<uint8_t> compressed(compressed_size);
        
        // Level 4: balanced speed and compression ratio
        size_t result = ZSTD_compress(compressed.data(), compressed_size, data, size, 4);
        
        if (ZSTD_isError(result)) {
            std::cerr << "Block compression failed: " << ZSTD_getErrorName(result) << std::endl;
            return std::vector<uint8_t>();
        }
        
        compressed.resize(result);
        return compressed;
    }

    // Decompress a single block with ZSTD
    static std::vector<uint8_t> decompress_block(const uint8_t* data, size_t compressed_size, 
                                                  size_t original_size) {
        std::vector<uint8_t> decompressed(original_size);
        
        size_t result = ZSTD_decompress(decompressed.data(), original_size, data, compressed_size);
        
        if (ZSTD_isError(result)) {
            std::cerr << "Block decompression failed: " << ZSTD_getErrorName(result) << std::endl;
            return std::vector<uint8_t>();
        }
        
        return decompressed;
    }

public:
    // Lossless compression: float32 -> uint32_t -> XOR-delta -> single-frame ZSTD MT
    static bool compress_lossless(const std::string& input_path, const std::string& output_path) {
        auto start = std::chrono::high_resolution_clock::now();

        std::ifstream input(input_path, std::ios::binary);
        if (!input) { std::cerr << "Cannot open input file: " << input_path << std::endl; return false; }
        input.seekg(0, std::ios::end);
        size_t file_size = input.tellg();
        input.seekg(0, std::ios::beg);

        std::vector<uint8_t> data(file_size);
        input.read(reinterpret_cast<char*>(data.data()), file_size);
        input.close();

        if (file_size < 8) { std::cerr << "File too small" << std::endl; return false; }
        uint64_t header_size;
        std::memcpy(&header_size, data.data(), sizeof(uint64_t));
        if (8 + header_size > file_size) { std::cerr << "Invalid header size" << std::endl; return false; }

        std::vector<uint8_t> header_data(data.begin(), data.begin() + 8 + header_size);
        std::vector<uint8_t> tensor_bytes(data.begin() + 8 + header_size, data.end());
        size_t num_floats = tensor_bytes.size() / sizeof(float);

        // Reinterpret tensor bytes as uint32_t words
        std::vector<uint32_t> u32(num_floats);
        std::memcpy(u32.data(), tensor_bytes.data(), tensor_bytes.size());

        // XOR-delta encode to create low-entropy residuals
        xor_delta_encode_inplace_u32(u32);

        // Compress entire tensor as single frame, multi-threaded
        unsigned int workers = std::thread::hardware_concurrency();
        if (workers == 0) workers = 4;
        auto compressed = zstd_compress_mt(reinterpret_cast<const uint8_t*>(u32.data()),
                                           u32.size() * sizeof(uint32_t),
                                           /*level*/ 10, /*workers*/ workers, /*LDM*/ true);
        if (compressed.empty()) return false;

        // Write output with one block
        std::ofstream output(output_path, std::ios::binary);
        if (!output) { std::cerr << "Cannot open output file" << std::endl; return false; }

        Header hdr;
        hdr.original_size = file_size;
        hdr.json_header_size = header_data.size();
        hdr.num_floats = static_cast<uint32_t>(num_floats);
        hdr.num_blocks = 1;
        hdr.compressed_tensor_size = compressed.size() + sizeof(BlockHeader);

        output.write(reinterpret_cast<const char*>(&hdr), sizeof(Header));
        output.write(reinterpret_cast<const char*>(header_data.data()), header_data.size());

        BlockHeader bh;
        bh.compressed_size = compressed.size();
        bh.original_size = u32.size() * sizeof(uint32_t);
        output.write(reinterpret_cast<const char*>(&bh), sizeof(BlockHeader));
        output.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
        output.close();

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        size_t output_size = sizeof(Header) + header_data.size() + sizeof(BlockHeader) + compressed.size();
        double ratio = static_cast<double>(file_size) / output_size;
        double speed_mbps = (file_size / (1024.0 * 1024.0)) / (dur.count() / 1000.0);

        std::cout << "\n=== Lossless Compression Results ===\n"
                  << "Original size:      " << file_size << " bytes\n"
                  << "Compressed size:    " << output_size << " bytes\n"
                  << "Compression ratio:  " << ratio << ":1\n"
                  << "Time:               " << dur.count() / 1000.0 << " s\n"
                  << "Speed:              " << speed_mbps << " MB/s\n"
                  << "Workers:            " << workers << std::endl;

        return true;
    }

    static bool decompress_lossless(const std::string& input_path, const std::string& output_path) {
        auto start = std::chrono::high_resolution_clock::now();

        std::ifstream input(input_path, std::ios::binary);
        if (!input) { std::cerr << "Cannot open input file" << std::endl; return false; }

        Header hdr;
        input.read(reinterpret_cast<char*>(&hdr), sizeof(Header));
        std::vector<uint8_t> header_data(hdr.json_header_size);
        input.read(reinterpret_cast<char*>(header_data.data()), hdr.json_header_size);

        // Single block expected
        BlockHeader bh;
        input.read(reinterpret_cast<char*>(&bh), sizeof(BlockHeader));
        std::vector<uint8_t> block(bh.compressed_size);
        input.read(reinterpret_cast<char*>(block.data()), bh.compressed_size);
        input.close();

        // Decompress and decode
        auto decompressed = zstd_decompress(block.data(), block.size(), bh.original_size);
        if (decompressed.empty()) return false;

        std::vector<uint32_t> u32(hdr.num_floats);
        std::memcpy(u32.data(), decompressed.data(), decompressed.size());
        xor_delta_decode_inplace_u32(u32);

        std::vector<uint8_t> tensor_bytes(hdr.num_floats * sizeof(float));
        std::memcpy(tensor_bytes.data(), u32.data(), tensor_bytes.size());

        std::ofstream output(output_path, std::ios::binary);
        if (!output) { std::cerr << "Cannot open output file" << std::endl; return false; }
        output.write(reinterpret_cast<const char*>(header_data.data()), header_data.size());
        output.write(reinterpret_cast<const char*>(tensor_bytes.data()), tensor_bytes.size());
        output.close();

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        size_t output_size = header_data.size() + tensor_bytes.size();
        double speed_mbps = (output_size / (1024.0 * 1024.0)) / (dur.count() / 1000.0);

        std::cout << "\n=== Lossless Decompression Results ===\n"
                  << "Decompressed size:  " << output_size << " bytes\n"
                  << "Time:               " << dur.count() / 1000.0 << " s\n"
                  << "Speed:              " << speed_mbps << " MB/s" << std::endl;
        return true;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Optimized LLM Codec for SafeTensors" << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "  Compress:   " << argv[0] << " -c <input.safetensors> <output.compressed>" << std::endl;
        std::cout << "  Decompress: " << argv[0] << " -d <input.compressed> <output.safetensors>" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    std::string input = argv[2];
    std::string output = argv[3];
    
    if (mode == "-c") {
        if (!OptimizedLLMCodec::compress_lossless(input, output)) {
            std::cerr << "Compression failed!" << std::endl;
            return 1;
        }
    } else if (mode == "-d") {
        if (!OptimizedLLMCodec::decompress_lossless(input, output)) {
            std::cerr << "Decompression failed!" << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Invalid mode. Use -c or -d" << std::endl;
        return 1;
    }
    
    return 0;
}