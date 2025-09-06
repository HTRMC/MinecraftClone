#pragma once
#include "VulkanContext.hpp"
#include "ShaderManager.hpp"
#include "DescriptorManager.hpp"
#include "MeshShaderPipeline.hpp"
#include "ChunkRenderer.hpp"
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Window;

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
    
    ChunkRenderer* getChunkRenderer() { return chunkRenderer.get(); }

private:
    void createSurface();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createFramebuffers();
    void createCommandBuffers();
    
    void recreateSwapChain();
    void cleanupSwapChain();
    
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    
    UniformBufferObject createUBO();

private:
    VulkanContext* vulkanContext;
    Window* window;
    
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    
    uint32_t currentFrame = 0;
    static const int MAX_FRAMES_IN_FLIGHT = 2;
    
    std::unique_ptr<ShaderManager> shaderManager;
    std::unique_ptr<DescriptorManager> descriptorManager;
    std::unique_ptr<MeshShaderPipeline> meshPipeline;
    std::unique_ptr<ChunkRenderer> chunkRenderer;
    
    bool initialized = false;
    bool framebufferResized = false;
};