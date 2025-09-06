#pragma once
#include "VulkanContext.hpp"
#include <string>
#include <unordered_map>
#include <vector>

struct TextureEntry {
    uint32_t textureId;
    std::string path;
    uint32_t width, height;
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;
};

class TextureManager {
public:
    TextureManager(VulkanContext* vulkanContext);
    ~TextureManager();
    
    void init();
    void cleanup();
    
    // Load a texture and return its ID for bindless access
    uint32_t loadTexture(const std::string& texturePath);
    
    // Get texture info by ID
    const TextureEntry* getTextureInfo(uint32_t textureId) const;
    
    // Get texture array resources for binding
    const std::vector<VkImageView>& getTextureImageViews() const { return textureImageViews; }
    VkSampler getTextureSampler() const { return textureSampler; }
    
    // Get texture count
    uint32_t getTextureCount() const { return static_cast<uint32_t>(textureEntries.size()); }
    
private:
    void createTextureSampler();
    void createTextureImage(const std::string& path, TextureEntry& entry);
    std::vector<uint8_t> loadPNG(const std::string& path, int& width, int& height);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    
private:
    VulkanContext* vulkanContext;
    
    // Texture tracking
    std::unordered_map<std::string, uint32_t> pathToTextureId;
    std::vector<TextureEntry> textureEntries;
    std::vector<VkImageView> textureImageViews; // For bindless descriptor array
    uint32_t nextTextureId = 0;
    
    // Vulkan resources
    VkSampler textureSampler = VK_NULL_HANDLE;
    
    bool initialized = false;
    static constexpr uint32_t MAX_TEXTURES = 4096;
};