#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>

// Palette-based implementation (similar to Minecraft)
// Pros: Memory efficient (especially for uniform chunks), scales well
// Cons: Slightly more complex, additional indirection on access

constexpr uint32_t CHUNK_SIZE_PALETTE = 16;
constexpr uint32_t BLOCKS_PER_CHUNK_PALETTE = CHUNK_SIZE_PALETTE * CHUNK_SIZE_PALETTE * CHUNK_SIZE_PALETTE;

enum class BlockTypePalette : uint16_t {
    AIR = 0,
    DIRT = 1,
    GRASS = 2,
    STONE = 3,
    WOOD = 4,
    LEAVES = 5,
    SAND = 6,
    WATER = 7,
    // Can support up to 65536 block types
};

class ChunkPalette {
private:
    // Palette: unique block types in this chunk
    std::vector<BlockTypePalette> palette;

    // Indices into palette
    // We use uint8_t by default (supports up to 256 unique blocks per chunk)
    // Could be upgraded to uint16_t if more variety is needed
    std::vector<uint8_t> indices;

public:
    ChunkPalette() {
        // Initialize with single air block
        palette.push_back(BlockTypePalette::AIR);
        indices.resize(BLOCKS_PER_CHUNK_PALETTE, 0);
    }

    // Get block at position (x, y, z)
    BlockTypePalette getBlock(uint32_t x, uint32_t y, uint32_t z) const {
        uint32_t index = y * CHUNK_SIZE_PALETTE * CHUNK_SIZE_PALETTE + z * CHUNK_SIZE_PALETTE + x;
        return palette[indices[index]];
    }

    // Set block at position (x, y, z)
    void setBlock(uint32_t x, uint32_t y, uint32_t z, BlockTypePalette type) {
        uint32_t linearIndex = y * CHUNK_SIZE_PALETTE * CHUNK_SIZE_PALETTE + z * CHUNK_SIZE_PALETTE + x;

        // Find or add block type to palette
        uint8_t paletteIndex = findOrAddToPalette(type);

        // Update index
        indices[linearIndex] = paletteIndex;
    }

    // Get block by linear index
    BlockTypePalette getBlockByIndex(uint32_t index) const {
        return palette[indices[index]];
    }

    // Set block by linear index
    void setBlockByIndex(uint32_t index, BlockTypePalette type) {
        uint8_t paletteIndex = findOrAddToPalette(type);
        indices[index] = paletteIndex;
    }

    // Check if chunk is empty (all air)
    bool isEmpty() const {
        return palette.size() == 1 && palette[0] == BlockTypePalette::AIR;
    }

    // Get palette size
    size_t getPaletteSize() const {
        return palette.size();
    }

    // Get memory usage in bytes
    size_t getMemoryUsage() const {
        return palette.size() * sizeof(BlockTypePalette) +
               indices.size() * sizeof(uint8_t);
    }

    // Get palette for inspection/debugging
    const std::vector<BlockTypePalette>& getPalette() const {
        return palette;
    }

    // Optimize palette by removing unused entries
    // Call this after bulk modifications
    void compactPalette() {
        // Count usage of each palette entry
        std::vector<uint32_t> usage(palette.size(), 0);
        for (uint8_t idx : indices) {
            usage[idx]++;
        }

        // Build new palette with only used entries
        std::vector<BlockTypePalette> newPalette;
        std::vector<uint8_t> remapping(palette.size());

        for (size_t i = 0; i < palette.size(); i++) {
            if (usage[i] > 0) {
                remapping[i] = static_cast<uint8_t>(newPalette.size());
                newPalette.push_back(palette[i]);
            }
        }

        // Remap indices
        for (uint8_t& idx : indices) {
            idx = remapping[idx];
        }

        palette = std::move(newPalette);
    }

    // Fill entire chunk with one block type
    void fill(BlockTypePalette type) {
        palette.clear();
        palette.push_back(type);
        std::fill(indices.begin(), indices.end(), 0);
    }

    // Export to simple array format (for GPU upload)
    std::vector<uint32_t> exportToArray() const {
        std::vector<uint32_t> result(BLOCKS_PER_CHUNK_PALETTE);
        for (size_t i = 0; i < BLOCKS_PER_CHUNK_PALETTE; i++) {
            result[i] = static_cast<uint32_t>(palette[indices[i]]);
        }
        return result;
    }

private:
    // Find block type in palette, or add it if not present
    uint8_t findOrAddToPalette(BlockTypePalette type) {
        // Search for existing entry
        for (size_t i = 0; i < palette.size(); i++) {
            if (palette[i] == type) {
                return static_cast<uint8_t>(i);
            }
        }

        // Add new entry
        if (palette.size() >= 256) {
            // Palette is full, need to compact or upgrade to uint16_t indices
            compactPalette();

            // Try again after compacting
            for (size_t i = 0; i < palette.size(); i++) {
                if (palette[i] == type) {
                    return static_cast<uint8_t>(i);
                }
            }

            // If still full, this is an error condition
            // In production, you'd want to handle this more gracefully
            throw std::runtime_error("Chunk palette overflow");
        }

        palette.push_back(type);
        return static_cast<uint8_t>(palette.size() - 1);
    }
};