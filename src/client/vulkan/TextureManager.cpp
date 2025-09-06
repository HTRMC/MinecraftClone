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
        if (entry.image != VK_NULL_HANDLE) {
            vkDestroyImage(device, entry.image, nullptr);
        }
        if (entry.imageMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, entry.imageMemory, nullptr);
        }
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
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open PNG file: " + path);
    }

    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    file.close();

    spng_ctx* ctx = spng_ctx_new(0);
    if (!ctx) {
        throw std::runtime_error("Failed to create SPNG context");
    }

    spng_set_png_buffer(ctx, data.data(), data.size());

    spng_ihdr ihdr;
    if (spng_get_ihdr(ctx, &ihdr) != 0) {
        spng_ctx_free(ctx);
        throw std::runtime_error("Failed to get PNG header");
    }

    width = ihdr.width;
    height = ihdr.height;

    size_t out_size;
    if (spng_decoded_image_size(ctx, SPNG_FMT_RGBA8, &out_size) != 0) {
        spng_ctx_free(ctx);
        throw std::runtime_error("Failed to get decoded image size");
    }

    std::vector<uint8_t> out(out_size);
    if (spng_decode_image(ctx, out.data(), out.size(), SPNG_FMT_RGBA8, 0) != 0) {
        spng_ctx_free(ctx);
        throw std::runtime_error("Failed to decode PNG image");
    }

    spng_ctx_free(ctx);
    return out;
}

void TextureManager::createTextureImage(const std::string& path, TextureEntry& entry) {
    int texWidth, texHeight;
    std::vector<uint8_t> pixels = loadPNG(path, texWidth, texHeight);
    
    entry.width = static_cast<uint32_t>(texWidth);
    entry.height = static_cast<uint32_t>(texHeight);
    
    VkDeviceSize imageSize = texWidth * texHeight * 4; // RGBA
    
    // Create staging buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(vulkanContext->getDevice(), &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create staging buffer");
    }
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vulkanContext->getDevice(), stagingBuffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkanContext->findMemoryType(memRequirements.memoryTypeBits, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    if (vkAllocateMemory(vulkanContext->getDevice(), &allocInfo, nullptr, &stagingBufferMemory) != VK_SUCCESS) {
        vkDestroyBuffer(vulkanContext->getDevice(), stagingBuffer, nullptr);
        throw std::runtime_error("Failed to allocate staging buffer memory");
    }
    
    vkBindBufferMemory(vulkanContext->getDevice(), stagingBuffer, stagingBufferMemory, 0);
    
    // Copy pixel data to staging buffer
    void* data;
    vkMapMemory(vulkanContext->getDevice(), stagingBufferMemory, 0, imageSize, 0, &data);
    std::memcpy(data, pixels.data(), static_cast<size_t>(imageSize));
    vkUnmapMemory(vulkanContext->getDevice(), stagingBufferMemory);
    
    // Create image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = entry.width;
    imageInfo.extent.height = entry.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateImage(vulkanContext->getDevice(), &imageInfo, nullptr, &entry.image) != VK_SUCCESS) {
        vkDestroyBuffer(vulkanContext->getDevice(), stagingBuffer, nullptr);
        vkFreeMemory(vulkanContext->getDevice(), stagingBufferMemory, nullptr);
        throw std::runtime_error("Failed to create texture image");
    }
    
    // Allocate image memory
    vkGetImageMemoryRequirements(vulkanContext->getDevice(), entry.image, &memRequirements);
    
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkanContext->findMemoryType(memRequirements.memoryTypeBits, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (vkAllocateMemory(vulkanContext->getDevice(), &allocInfo, nullptr, &entry.imageMemory) != VK_SUCCESS) {
        vkDestroyImage(vulkanContext->getDevice(), entry.image, nullptr);
        vkDestroyBuffer(vulkanContext->getDevice(), stagingBuffer, nullptr);
        vkFreeMemory(vulkanContext->getDevice(), stagingBufferMemory, nullptr);
        throw std::runtime_error("Failed to allocate texture image memory");
    }
    
    vkBindImageMemory(vulkanContext->getDevice(), entry.image, entry.imageMemory, 0);
    
    // Transition image layout and copy buffer to image
    transitionImageLayout(entry.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, entry.image, entry.width, entry.height);
    transitionImageLayout(entry.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    
    // Cleanup staging buffer
    vkDestroyBuffer(vulkanContext->getDevice(), stagingBuffer, nullptr);
    vkFreeMemory(vulkanContext->getDevice(), stagingBufferMemory, nullptr);
    
    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = entry.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    if (vkCreateImageView(vulkanContext->getDevice(), &viewInfo, nullptr, &entry.imageView) != VK_SUCCESS) {
        vkDestroyImage(vulkanContext->getDevice(), entry.image, nullptr);
        vkFreeMemory(vulkanContext->getDevice(), entry.imageMemory, nullptr);
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

void TextureManager::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::invalid_argument("Unsupported layout transition!");
    }
    
    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    
    endSingleTimeCommands(commandBuffer);
}

void TextureManager::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    
    endSingleTimeCommands(commandBuffer);
}

VkCommandBuffer TextureManager::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = vulkanContext->getCommandPool();
    allocInfo.commandBufferCount = 1;
    
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(vulkanContext->getDevice(), &allocInfo, &commandBuffer);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void TextureManager::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(vulkanContext->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(vulkanContext->getGraphicsQueue());
    
    vkFreeCommandBuffers(vulkanContext->getDevice(), vulkanContext->getCommandPool(), 1, &commandBuffer);
}