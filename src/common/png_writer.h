#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifndef RT_NO_ZLIB
#include <zlib.h>
#endif

namespace png_detail {
inline uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1u)) : (c >> 1u);
            }
            table[i] = c;
        }
        init = true;
    }
    crc ^= 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8u);
    }
    return crc ^ 0xFFFFFFFFu;
}

inline void write_be32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

inline void append_chunk(std::vector<uint8_t>& out, const char type[4], const uint8_t* data, uint32_t len) {
    write_be32(out, len);
    size_t type_pos = out.size();
    out.insert(out.end(), type, type + 4);
    if (len > 0 && data != nullptr) {
        out.insert(out.end(), data, data + len);
    }
    uint32_t crc = crc32_update(0u, out.data() + type_pos, 4u + len);
    write_be32(out, crc);
}

inline uint32_t adler32_calc(const uint8_t* data, size_t len) {
    uint32_t a = 1u, b = 0u;
    const uint32_t mod = 65521u;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % mod;
        b = (b + a) % mod;
    }
    return (b << 16) | a;
}

inline void hsv_to_rgb(float h_deg, float s, float v, uint8_t& r, uint8_t& g, uint8_t& b) {
    float h = fmodf(h_deg, 360.0f);
    if (h < 0.0f) h += 360.0f;
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rf = 0, gf = 0, bf = 0;
    if (h < 60.0f)      { rf = c; gf = x; bf = 0; }
    else if (h < 120.0f){ rf = x; gf = c; bf = 0; }
    else if (h < 180.0f){ rf = 0; gf = c; bf = x; }
    else if (h < 240.0f){ rf = 0; gf = x; bf = c; }
    else if (h < 300.0f){ rf = x; gf = 0; bf = c; }
    else                { rf = c; gf = 0; bf = x; }
    r = static_cast<uint8_t>((rf + m) * 255.0f + 0.5f);
    g = static_cast<uint8_t>((gf + m) * 255.0f + 0.5f);
    b = static_cast<uint8_t>((bf + m) * 255.0f + 0.5f);
}
} // namespace png_detail

inline bool write_image_file(const std::string& path, int width, int height, const uint8_t* rgb) {
    if (path.empty() || width <= 0 || height <= 0 || !rgb) return false;

    if (path.size() >= 4 && path.substr(path.size() - 4) == ".ppm") {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        std::string hdr = "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
        f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        f.write(reinterpret_cast<const char*>(rgb), static_cast<std::streamsize>(width) * height * 3);
        return f.good();
    }

    const size_t stride = static_cast<size_t>(width) * 3;
    std::vector<uint8_t> raw(static_cast<size_t>(height) * (stride + 1));

#ifndef RT_NO_ZLIB
    // PNG Filter type 1 (Sub) for high compression efficiency
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = rgb + static_cast<size_t>(y) * stride;
        uint8_t* dst = raw.data() + static_cast<size_t>(y) * (stride + 1);
        dst[0] = 1;
        for (size_t i = 0; i < stride; ++i) {
            uint8_t left = (i >= 3) ? row[i - 3] : 0;
            dst[1 + i] = static_cast<uint8_t>(row[i] - left);
        }
    }
    uLongf comp_bound = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> idat(comp_bound);
    if (compress2(idat.data(), &comp_bound, raw.data(), static_cast<uLong>(raw.size()), 6) != Z_OK) {
        return false;
    }
    idat.resize(comp_bound);
#else
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = rgb + static_cast<size_t>(y) * stride;
        uint8_t* dst = raw.data() + static_cast<size_t>(y) * (stride + 1);
        dst[0] = 0;
        std::copy(row, row + stride, dst + 1);
    }
    std::vector<uint8_t> idat;
    idat.push_back(0x78);
    idat.push_back(0x01);
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t chunk = raw.size() - pos;
        if (chunk > 65535u) chunk = 65535u;
        bool last = (pos + chunk == raw.size());
        idat.push_back(last ? 0x01 : 0x00);
        uint16_t len = static_cast<uint16_t>(chunk);
        uint16_t nlen = static_cast<uint16_t>(~len);
        idat.push_back(static_cast<uint8_t>(len & 0xFF));
        idat.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
        idat.push_back(static_cast<uint8_t>(nlen & 0xFF));
        idat.push_back(static_cast<uint8_t>((nlen >> 8) & 0xFF));
        idat.insert(idat.end(), raw.begin() + pos, raw.begin() + pos + chunk);
        pos += chunk;
    }
    uint32_t adl = png_detail::adler32_calc(raw.data(), raw.size());
    png_detail::write_be32(idat, adl);
#endif

    std::vector<uint8_t> png;
    png.reserve(idat.size() + 128);
    const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    png.insert(png.end(), sig, sig + 8);

    std::vector<uint8_t> ihdr;
    png_detail::write_be32(ihdr, static_cast<uint32_t>(width));
    png_detail::write_be32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8); // bit depth
    ihdr.push_back(2); // color type: RGB
    ihdr.push_back(0); // compression
    ihdr.push_back(0); // filter
    ihdr.push_back(0); // interlace
    png_detail::append_chunk(png, "IHDR", ihdr.data(), static_cast<uint32_t>(ihdr.size()));
    png_detail::append_chunk(png, "IDAT", idat.data(), static_cast<uint32_t>(idat.size()));
    png_detail::append_chunk(png, "IEND", nullptr, 0);

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return out.good();
}

inline bool write_heatmap_png(
    const std::string& path,
    int width,
    int height,
    int tile_size,
    const std::vector<int>& tile_rank,
    const std::vector<int>& tile_thread,
    int num_ranks,
    int num_threads
) {
    if (path.empty() || width <= 0 || height <= 0) return false;
    int tiles_x = (width + tile_size - 1) / tile_size;
    int tiles_y = (height + tile_size - 1) / tile_size;
    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3, 0);

    for (int y = 0; y < height; ++y) {
        int ty = y / tile_size;
        if (ty >= tiles_y) ty = tiles_y - 1;
        bool y_edge = (y % tile_size == 0);
        for (int x = 0; x < width; ++x) {
            int tx = x / tile_size;
            if (tx >= tiles_x) tx = tiles_x - 1;
            int tidx = ty * tiles_x + tx;
            int r_id = (tidx < static_cast<int>(tile_rank.size())) ? tile_rank[tidx] : 0;
            int t_id = (tidx < static_cast<int>(tile_thread.size())) ? tile_thread[tidx] : 0;

            float hue = 0.0f;
            float sat = 0.78f;
            float val = 0.88f;

            if (num_ranks > 1) {
                hue = fmodf(static_cast<float>(r_id) * 137.508f + 18.0f, 360.0f);
                if (num_threads > 1) {
                    float frac = static_cast<float>(t_id % num_threads) / static_cast<float>(num_threads);
                    val = 0.52f + 0.44f * frac;
                    sat = 0.72f + 0.20f * (1.0f - frac);
                }
            } else if (num_threads > 1) {
                hue = fmodf(static_cast<float>(t_id) * 137.508f + 25.0f, 360.0f);
                sat = 0.78f;
                val = 0.90f;
            } else {
                hue = 205.0f;
                sat = 0.65f;
                val = 0.82f;
            }

            uint8_t cr = 0, cg = 0, cb = 0;
            png_detail::hsv_to_rgb(hue, sat, val, cr, cg, cb);

            bool x_edge = (x % tile_size == 0);
            if (x_edge || y_edge) {
                cr = static_cast<uint8_t>(cr * 0.45f);
                cg = static_cast<uint8_t>(cg * 0.45f);
                cb = static_cast<uint8_t>(cb * 0.45f);
            }

            size_t off = (static_cast<size_t>(y) * width + x) * 3;
            rgb[off] = cr;
            rgb[off + 1] = cg;
            rgb[off + 2] = cb;
        }
    }

    return write_image_file(path, width, height, rgb.data());
}
