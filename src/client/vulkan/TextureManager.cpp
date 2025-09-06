#include "TextureManager.hpp"
#include "Logger.hpp"
#include <spng.h>
#include <fstream>
#include <stdexcept>
#include <cstring>

TextureManager::TextureManager(VulkanContext* vulkanContext) 
    : vulkanContext(vulkanContext) {
    textureEntries.reserve(MAX_TEXTURES);
    textureImageViews.reserve(MAX_TEXTURES);
}

TextureManager::~TextureManager() {
    cleanup();
}

void TextureManager::init() {
    if (initialized) return;
    
    createTextureSampler();
    
    initialized = true;
    Logger::info("TextureManager", "Initialized bindless texture system");
}

void TextureManager::cleanup() {
    if (!initialized) return;
    
    VkDevice device = vulkanContext->getDevice();
    
    // Cleanup all textures
    for (auto& entry : textureEntries) {
        if (entry.imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, entry.imageView, nullptr);
        }
        vulkanContext->destroyImage(entry.imageInfo);
    }
    
    if (textureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, textureSampler, nullptr);
        textureSampler = VK_NULL_HANDLE;
    }
    
    textureEntries.clear();
    textureImageViews.clear();
    pathToTextureId.clear();
    
    initialized = false;
}

uint32_t TextureManager::loadTexture(const std::string& texturePath) {
    // Check if already loaded
    auto it = pathToTextureId.find(texturePath);
    if (it != pathToTextureId.end()) {
        return it->second;
    }
    
    // Check texture limit
    if (nextTextureId >= MAX_TEXTURES) {
        Logger::error("TextureManager", "Maximum texture limit reached: " + std::to_string(MAX_TEXTURES));
        return 0; // Return default texture ID
    }
    
    // Create new texture entry
    TextureEntry entry;
    entry.textureId = nextTextureId;
    entry.path = texturePath;
    
    try {
        createTextureImage(texturePath, entry);
        
        // Add to tracking
        textureEntries.push_back(entry);
        textureImageViews.push_back(entry.imageView);
        pathToTextureId[texturePath] = nextTextureId;
        
        Logger::info("TextureManager", "Loaded texture " + std::to_string(nextTextureId) + ": " + texturePath);
        
        return nextTextureId++;
    } catch (const std::exception& e) {
        Logger::error("TextureManager", "Failed to load texture " + texturePath + ": " + std::string(e.what()));
        return 0; // Return default texture ID
    }
}

const TextureEntry* TextureManager::getTextureInfo(uint32_t textureId) const {
    if (textureId < textureEntries.size()) {
        return &textureEntries[textureId];
    }
    return nullptr;
}

std::vector<uint8_t> TextureManager::loadPNG(const std::string& path, int& width, int& height) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Failed to open PNG file: " + path);
    }
    
    spng_ctx* ctx = spng_ctx_new(0);
    if (!ctx) {
        fclose(file);
        throw std::runtime_error("Failed to create SPNG context");
    }
    
    spng_set_png_file(ctx, file);
    
    spng_ihdr ihdr;
    int ret = spng_get_ihdr(ctx, &ihdr);
    if (ret) {
        spng_ctx_free(ctx);
        fclose(file);
        throw std::runtime_error("Failed to get PNG header: " + std::string(spng_strerror(ret)));
    }
    
    width = ihdr.width;
    height = ihdr.height;
    
    Logger::debug("TextureManager", "PNG info: " + std::to_string(ihdr.width) + "x" + std::to_string(ihdr.height) + 
                  ", bit_depth=" + std::to_string(ihdr.bit_depth) + ", color_type=" + std::to_string(ihdr.color_type));
    
    int output_format = SPNG_FMT_RGBA8;
    int decode_flags = SPNG_DECODE_TRNS;
    
    size_t out_size;
    ret = spng_decoded_image_size(ctx, output_format, &out_size);
    if (ret) {
        spng_ctx_free(ctx);
        fclose(file);
        throw std::runtime_error("Failed to get decoded image size: " + std::string(spng_strerror(ret)));
    }
    
    std::vector<uint8_t> out(out_size);
    ret = spng_decode_image(ctx, out.data(), out_size, output_format, decode_flags);
    if (ret) {
        spng_ctx_free(ctx);
        fclose(file);
        throw std::runtime_error("Failed to decode PNG image: " + std::string(spng_strerror(ret)));
    }
    
    spng_ctx_free(ctx);
    fclose(file);
    
    Logger::info("TextureManager", "Loaded PNG: " + path + " (" + std::to_string(width) + "x" + std::to_string(height) + ")");
    return out;
}

void TextureManager::createTextureImage(const std::string& path, TextureEntry& entry) {
    int texWidth, texHeight;
    std::vector<uint8_t> pixels = loadPNG(path, texWidth, texHeight);
    
    entry.width = static_cast<uint32_t>(texWidth);
    entry.height = static_cast<uint32_t>(texHeight);
    
    VkDeviceSize imageSize = texWidth * texHeight * 4; // RGBA
    
    // Create staging buffer using VMA
    BufferInfo stagingBuffer = vulkanContext->createStagingBuffer(imageSize);
    
    // Copy pixel data to staging buffer
    void* data = vulkanContext->mapBuffer(stagingBuffer);
    std::memcpy(data, pixels.data(), static_cast<size_t>(imageSize));
    vulkanContext->unmapBuffer(stagingBuffer);
    
    // Create image using VMA
    entry.imageInfo = vulkanContext->createImage(entry.width, entry.height, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
        VMA_MEMORY_USAGE_GPU_ONLY);
    
    // Transition image layout and copy buffer to image
    vulkanContext->transitionImageLayout(entry.imageInfo.image, VK_FORMAT_R8G8B8A8_SRGB, 
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vulkanContext->copyBufferToImage(stagingBuffer.buffer, entry.imageInfo.image, entry.width, entry.height);
    vulkanContext->transitionImageLayout(entry.imageInfo.image, VK_FORMAT_R8G8B8A8_SRGB, 
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    
    // Cleanup staging buffer
    vulkanContext->destroyBuffer(stagingBuffer);
    
    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = entry.imageInfo.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    if (vkCreateImageView(vulkanContext->getDevice(), &viewInfo, nullptr, &entry.imageView) != VK_SUCCESS) {
        vulkanContext->destroyImage(entry.imageInfo);
        throw std::runtime_error("Failed to create texture image view");
    }
}

void TextureManager::createTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST; // Pixel-perfect filtering for block textures
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    
    if (vkCreateSampler(vulkanContext->getDevice(), &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture sampler");
    }
}

