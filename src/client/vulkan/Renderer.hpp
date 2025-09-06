#pragma once
#include "VulkanContext.hpp"
#include "ShaderManager.hpp"
#include "DescriptorManager.hpp"
#include "MeshShaderPipeline.hpp"
#include "TextureManager.hpp"
#include "ChunkRenderer.hpp"
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Window;
class Camera;

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class Renderer {
public:
    Renderer(VulkanContext* vulkanContext, Window* window);
    ~Renderer();
    
    void init();
    void cleanup();
    void render();
    void setCamera(Camera* camera) { this->camera = camera; }
    
    ChunkRenderer* getChunkRenderer() { return chunkRenderer.get(); }

private:
    void createSurface();
    void createSwapChain();
    void createImageViews();
    void createDepthResources();
    void createDynamicRenderingInfo();
    void createCommandBuffers();
    
    void recreateSwapChain();
    void cleanupSwapChain();
    
    VkFormat findDepthFormat();
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
    bool hasStencilComponent(VkFormat format);
    
    void recordDynamicRenderingCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    
    UniformBufferObject createUBO();

private:
    VulkanContext* vulkanContext;
    Window* window;
    Camera* camera = nullptr;
    
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    
    std::vector<VkCommandBuffer> commandBuffers;
    
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    
    // Per-swapchain-image semaphores for proper synchronization
    std::vector<VkSemaphore> perImageRenderFinishedSemaphores;
    
    // Timeline semaphores for advanced synchronization
    TimelineSemaphore frameTimelineSemaphore;
    
    uint32_t currentFrame = 0;
    uint64_t frameNumber = 0;
    static const int MAX_FRAMES_IN_FLIGHT = 2;
    
    std::unique_ptr<ShaderManager> shaderManager;
    std::unique_ptr<DescriptorManager> descriptorManager;
    std::unique_ptr<TextureManager> textureManager;
    std::unique_ptr<MeshShaderPipeline> meshPipeline;
    std::unique_ptr<ChunkRenderer> chunkRenderer;
    
    bool initialized = false;
    bool framebufferResized = false;
    std::vector<bool> commandBuffersRecorded;
    std::vector<uint32_t> recordedForImageIndex;
    std::vector<uint64_t> lastSubmittedFrame;
    
    VkRenderingInfo dynamicRenderingInfo = {};
};