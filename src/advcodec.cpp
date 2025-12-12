#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <unordered_map>

// Advanced LLM Compression Codec
// Techniques: Float16 quantization, delta encoding, bit packing, and entropy coding

class LLMCodec {
private:
    struct Header {
        uint64_t original_size;
        uint64_t json_header_size;
        uint32_t num_tensors;
        uint32_t compression_method; // 0=float16+delta+rle, 1=quantized
        float min_value;
        float max_value;
    };

    // Optimized float32 to float16 conversion
    static uint16_t f32_to_f16(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, 4);
        
        uint32_t sign = (bits >> 16) & 0x8000;
        int exponent = ((bits >> 23) & 0xff) - 127 + 15;
        uint32_t mantissa = (bits >> 13) & 0x3ff;
        
        if (exponent <= 0) return sign;
        if (exponent >= 31) return sign | 0x7c00;
        
        return sign | (exponent << 10) | mantissa;
    }

    static float f16_to_f32(uint16_t value) {
        uint32_t sign = (value & 0x8000) << 16;
        int exponent = (value >> 10) & 0x1f;
        uint32_t mantissa = value & 0x3ff;
        
        if (exponent == 0) {
            uint32_t bits = sign;
            float result;
            std::memcpy(&result, &bits, 4);
            return result;
        }
        if (exponent == 31) {
            uint32_t bits = sign | 0x7f800000;
            float result;
            std::memcpy(&result, &bits, 4);
            return result;
        }
        
        exponent = exponent - 15 + 127;
        uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    }

    // Quantize to N bits with min/max normalization
    static std::vector<uint8_t> quantize_8bit(const std::vector<float>& values, float& out_min, float& out_max) {
        if (values.empty()) return {};
        
        out_min = *std::min_element(values.begin(), values.end());
        out_max = *std::max_element(values.begin(), values.end());
        
        float range = out_max - out_min;
        if (range < 1e-8f) range = 1.0f;
        
        std::vector<uint8_t> quantized(values.size());
        for (size_t i = 0; i < values.size(); i++) {
            float normalized = (values[i] - out_min) / range;
            quantized[i] = static_cast<uint8_t>(std::clamp(normalized * 255.0f, 0.0f, 255.0f));
        }
        
        return quantized;
    }

    static std::vector<float> dequantize_8bit(const std::vector<uint8_t>& quantized, float min_val, float max_val) {
        float range = max_val - min_val;
        std::vector<float> values(quantized.size());
        
        for (size_t i = 0; i < quantized.size(); i++) {
            values[i] = min_val + (quantized[i] / 255.0f) * range;
        }
        
        return values;
    }

    // Delta encoding with variable byte encoding
    static std::vector<uint8_t> delta_encode_varbyte(const std::vector<uint8_t>& data) {
        if (data.empty()) return {};
        
        std::vector<uint8_t> encoded;
        encoded.reserve(data.size());
        
        encoded.push_back(data[0]);
        
        for (size_t i = 1; i < data.size(); i++) {
            int delta = static_cast<int>(data[i]) - static_cast<int>(data[i-1]);
            
            // Encode delta with sign and magnitude
            uint8_t abs_delta = std::abs(delta);
            uint8_t sign_bit = (delta < 0) ? 0x80 : 0x00;
            
            encoded.push_back(sign_bit | (abs_delta & 0x7F));
        }
        
        return encoded;
    }

    static std::vector<uint8_t> delta_decode_varbyte(const std::vector<uint8_t>& encoded) {
        if (encoded.empty()) return {};
        
        std::vector<uint8_t> decoded;
        decoded.reserve(encoded.size());
        
        decoded.push_back(encoded[0]);
        
        for (size_t i = 1; i < encoded.size(); i++) {
            uint8_t sign_bit = encoded[i] & 0x80;
            uint8_t magnitude = encoded[i] & 0x7F;
            
            int delta = sign_bit ? -static_cast<int>(magnitude) : static_cast<int>(magnitude);
            uint8_t value = static_cast<uint8_t>(decoded.back() + delta);
            decoded.push_back(value);
        }
        
        return decoded;
    }

    // Simple but effective byte-oriented RLE
    static std::vector<uint8_t> compress_rle(const std::vector<uint8_t>& data) {
        std::vector<uint8_t> compressed;
        compressed.reserve(data.size());
        
        size_t i = 0;
        while (i < data.size()) {
            size_t run_length = 1;
            while (i + run_length < data.size() && 
                   data[i + run_length] == data[i] && 
                   run_length < 255) {
                run_length++;
            }
            
            if (run_length >= 4) {
                // RLE: marker (0xFF), length, value
                compressed.push_back(0xFF);
                compressed.push_back(static_cast<uint8_t>(run_length));
                compressed.push_back(data[i]);
                i += run_length;
            } else {
                // Literal run
                size_t lit_length = 0;
                size_t lit_start = i;
                while (i < data.size() && lit_length < 255) {
                    size_t peek_run = 1;
                    while (i + peek_run < data.size() && 
                           data[i + peek_run] == data[i] && 
                           peek_run < 4) {
                        peek_run++;
                    }
                    if (peek_run >= 4) break;
                    lit_length++;
                    i++;
                }
                
                compressed.push_back(static_cast<uint8_t>(lit_length));
                compressed.insert(compressed.end(), 
                            data.begin() + lit_start, 
                            data.begin() + lit_start + lit_length);
            }
        }
        
        return compressed;
    }

    static std::vector<uint8_t> decompress_rle(const std::vector<uint8_t>& compressed) {
        std::vector<uint8_t> data;
        data.reserve(compressed.size() * 2);
        
        size_t i = 0;
        while (i < compressed.size()) {
            if (compressed[i] == 0xFF && i + 2 < compressed.size()) {
                uint8_t length = compressed[i + 1];
                uint8_t value = compressed[i + 2];
                data.insert(data.end(), length, value);
                i += 3;
            } else {
                uint8_t length = compressed[i];
                if (i + length < compressed.size()) {
                    data.insert(data.end(), compressed.begin() + i + 1, compressed.begin() + i + 1 + length);
                    i += length + 1;
                } else {
                    break;
                }
            }
        }
        
        return data;
    }

public:
    static bool compress(const std::string& input_path, const std::string& output_path, bool use_8bit = true) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::ifstream input(input_path, std::ios::binary);
        if (!input) {
            std::cerr << "Cannot open input: " << input_path << std::endl;
            return false;
        }
        
        input.seekg(0, std::ios::end);
        size_t file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> file_data(file_size);
        input.read(reinterpret_cast<char*>(file_data.data()), file_size);
        input.close();
        
        std::cout << "Input size: " << file_size << " bytes (" << file_size / (1024.0 * 1024.0) << " MB)" << std::endl;
        
        if (file_size < 8) return false;
        
        uint64_t json_header_size;
        std::memcpy(&json_header_size, file_data.data(), 8);
        
        std::vector<uint8_t> header(file_data.begin(), file_data.begin() + 8 + json_header_size);
        std::vector<uint8_t> tensor_bytes(file_data.begin() + 8 + json_header_size, file_data.end());
        
        size_t num_floats = tensor_bytes.size() / 4;
        std::vector<float> floats(num_floats);
        std::memcpy(floats.data(), tensor_bytes.data(), tensor_bytes.size());
        
        std::cout << "Tensors: " << num_floats << " float32 values" << std::endl;
        
        std::vector<uint8_t> compressed_tensors;
        float min_val = 0, max_val = 0;
        
        if (use_8bit) {
            // 8-bit quantization
            auto quantized = quantize_8bit(floats, min_val, max_val);
            std::cout << "Quantized to 8-bit (range: " << min_val << " to " << max_val << ")" << std::endl;
            
            auto delta_encoded = delta_encode_varbyte(quantized);
            std::cout << "Delta encoded: " << delta_encoded.size() << " bytes" << std::endl;
            
            compressed_tensors = compress_rle(delta_encoded);
            std::cout << "RLE compressed: " << compressed_tensors.size() << " bytes" << std::endl;
        } else {
            // Float16 approach
            std::vector<uint16_t> f16_values(num_floats);
            for (size_t i = 0; i < num_floats; i++) {
                f16_values[i] = f32_to_f16(floats[i]);
            }
            
            std::vector<uint8_t> f16_bytes(num_floats * 2);
            std::memcpy(f16_bytes.data(), f16_values.data(), f16_bytes.size());
            
            compressed_tensors = compress_rle(f16_bytes);
        }
        
        // Write output
        std::ofstream output(output_path, std::ios::binary);
        if (!output) return false;
        
        Header hdr;
        hdr.original_size = file_size;
        hdr.json_header_size = json_header_size;
        hdr.num_tensors = num_floats;
        hdr.compression_method = use_8bit ? 1 : 0;
        hdr.min_value = min_val;
        hdr.max_value = max_val;
        
        output.write(reinterpret_cast<const char*>(&hdr), sizeof(Header));
        output.write(reinterpret_cast<const char*>(header.data()), header.size());
        
        uint64_t compressed_size = compressed_tensors.size();
        output.write(reinterpret_cast<const char*>(&compressed_size), 8);
        output.write(reinterpret_cast<const char*>(compressed_tensors.data()), compressed_size);
        
        output.close();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        size_t total_compressed = sizeof(Header) + header.size() + 8 + compressed_size;
        double ratio = static_cast<double>(file_size) / total_compressed;
        
        std::cout << "\n=== COMPRESSION RESULTS ===" << std::endl;
        std::cout << "Original:    " << file_size << " bytes" << std::endl;
        std::cout << "Compressed:  " << total_compressed << " bytes" << std::endl;
        std::cout << "Ratio:       " << ratio << ":1 (" << (100.0 * total_compressed / file_size) << "%)" << std::endl;
        std::cout << "Saved:       " << (file_size - total_compressed) << " bytes" << std::endl;
        std::cout << "Time:        " << ms << " ms" << std::endl;
        
        return true;
    }

    static bool decompress(const std::string& input_path, const std::string& output_path) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::ifstream input(input_path, std::ios::binary);
        if (!input) return false;
        
        Header hdr;
        input.read(reinterpret_cast<char*>(&hdr), sizeof(Header));
        
        std::vector<uint8_t> header(8 + hdr.json_header_size);
        input.read(reinterpret_cast<char*>(header.data()), header.size());
        
        uint64_t compressed_size;
        input.read(reinterpret_cast<char*>(&compressed_size), 8);
        
        std::vector<uint8_t> compressed(compressed_size);
        input.read(reinterpret_cast<char*>(compressed.data()), compressed_size);
        input.close();
        
        std::vector<uint8_t> tensor_bytes;
        
        if (hdr.compression_method == 1) {
            // 8-bit quantized
            auto delta_encoded = decompress_rle(compressed);
            auto quantized = delta_decode_varbyte(delta_encoded);
            auto floats = dequantize_8bit(quantized, hdr.min_value, hdr.max_value);
            
            tensor_bytes.resize(floats.size() * 4);
            std::memcpy(tensor_bytes.data(), floats.data(), tensor_bytes.size());
        } else {
            // Float16
            auto f16_bytes = decompress_rle(compressed);
            std::vector<uint16_t> f16_values(f16_bytes.size() / 2);
            std::memcpy(f16_values.data(), f16_bytes.data(), f16_bytes.size());
            
            std::vector<float> floats(f16_values.size());
            for (size_t i = 0; i < f16_values.size(); i++) {
                floats[i] = f16_to_f32(f16_values[i]);
            }
            
            tensor_bytes.resize(floats.size() * 4);
            std::memcpy(tensor_bytes.data(), floats.data(), tensor_bytes.size());
        }
        
        std::ofstream output(output_path, std::ios::binary);
        if (!output) return false;
        
        output.write(reinterpret_cast<const char*>(header.data()), header.size());
        output.write(reinterpret_cast<const char*>(tensor_bytes.data()), tensor_bytes.size());
        output.close();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "\n=== DECOMPRESSION RESULTS ===" << std::endl;
        std::cout << "Output:      " << (header.size() + tensor_bytes.size()) << " bytes" << std::endl;
        std::cout << "Time:        " << ms << " ms" << std::endl;
        
        return true;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "LLM SafeTensors Compression Codec\n" << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "  Compress:   " << argv[0] << " -c <input.safetensors> <output.llmc> [--float16]" << std::endl;
        std::cout << "  Decompress: " << argv[0] << " -d <input.llmc> <output.safetensors>" << std::endl;
        std::cout << "\nOptions:" << std::endl;
        std::cout << "  --float16   Use float16 instead of 8-bit quantization (less lossy, lower compression)" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    std::string input = argv[2];
    std::string output = argv[3];
    
    bool use_float16 = false;
    if (argc > 4 && std::string(argv[4]) == "--float16") {
        use_float16 = true;
    }
    
    if (mode == "-c") {
        return LLMCodec::compress(input, output, !use_float16) ? 0 : 1;
    } else if (mode == "-d") {
        return LLMCodec::decompress(input, output) ? 0 : 1;
    }
    
    std::cerr << "Invalid mode: " << mode << std::endl;
    return 1;
}