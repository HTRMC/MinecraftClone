#pragma once
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <cstdint>

// UBO for matrices (binding = 0)
struct alignas(16) UniformBufferObject {
    glm::mat4 view;
    glm::mat4 proj;
};

struct alignas(4) FaceData {
    uint32_t positionAndFlags;     // x(5) + y(5) + z(5) + isBackFace(1) + lightIndex(16)
    uint32_t blockAndQuad;         // texture(16) + quadIndex(16)
    uint32_t chunkIndex;           // chunk index
    
    // Utility functions for packing/unpacking
    static uint32_t packPosition(uint32_t x, uint32_t y, uint32_t z, bool isBackFace, uint16_t lightIndex) {
        return (x & 0x1F) | 
               ((y & 0x1F) << 5) | 
               ((z & 0x1F) << 10) | 
               (isBackFace ? (1u << 15) : 0) | 
               (static_cast<uint32_t>(lightIndex) << 16);
    }
    
    static uint32_t packBlockAndQuad(uint16_t texture, uint16_t quadIndex) {
        return (static_cast<uint32_t>(texture) << 16) | static_cast<uint32_t>(quadIndex);
    }
    
    void setPosition(uint32_t x, uint32_t y, uint32_t z, bool isBackFace, uint16_t lightIndex) {
        positionAndFlags = packPosition(x, y, z, isBackFace, lightIndex);
    }
    
    void setBlockAndQuad(uint16_t texture, uint16_t quadIndex) {
        blockAndQuad = packBlockAndQuad(texture, quadIndex);
    }
};
static_assert(sizeof(FaceData) == 12, "FaceData must be 12 bytes");

struct alignas(16) ModelData {
    glm::vec4 vertices[4];    // 64 bytes - quad vertices in unit cube space
    glm::vec2 uvCoords[4];    // 32 bytes - UV coordinates
    glm::vec4 faceNormal;     // 16 bytes - face normal
};
static_assert(sizeof(ModelData) == 112, "ModelData must be 112 bytes");

struct alignas(16) LightData {
    uint32_t vertex0;  // ao(5) + skyLight(5) + blockLight(5) + r(5) + g(5) + b(5) + padding(2)
    uint32_t vertex1;  // Same layout for vertex 1
    uint32_t vertex2;  // Same layout for vertex 2  
    uint32_t vertex3;  // Same layout for vertex 3
    
    static uint32_t packVertexLight(uint8_t ao, uint8_t skyLight, uint8_t blockLight, 
                                   uint8_t r, uint8_t g, uint8_t b) {
        return (ao & 0x1F) | 
               ((skyLight & 0x1F) << 5) | 
               ((blockLight & 0x1F) << 10) | 
               ((r & 0x1F) << 15) | 
               ((g & 0x1F) << 20) | 
               ((b & 0x1F) << 25);
    }
    
    void setVertexLighting(int vertex, uint8_t ao, uint8_t skyLight, uint8_t blockLight, 
                          uint8_t r, uint8_t g, uint8_t b) {
        uint32_t packed = packVertexLight(ao, skyLight, blockLight, r, g, b);
        switch (vertex) {
            case 0: vertex0 = packed; break;
            case 1: vertex1 = packed; break;
            case 2: vertex2 = packed; break;
            case 3: vertex3 = packed; break;
        }
    }
};
static_assert(sizeof(LightData) == 16, "LightData must be 16 bytes");

// Descriptor set bindings
enum class DescriptorBinding : uint32_t {
    UBO = 0,           // Uniform buffer (matrices)
    FACE_DATA = 1,     // Storage buffer
    MODEL_DATA = 2,    // Storage buffer
    LIGHT_DATA = 3     // Storage buffer
};