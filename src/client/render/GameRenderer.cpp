#include "GameRenderer.hpp"

#include <algorithm>

float GameRenderer::getFarPlaneDistance() {
    return std::max(viewDistanceBlocks * 4.0f,  static_cast<float>(gameOptions.getCloudRenderDistance().getValue() * 16));
}

float GameRenderer::getFov() {
    // Default FOV is 70 degrees in Minecraft
    // Note: In full Minecraft, this would also account for:
    // - FOV settings (default 70, range 30-110)
    // - FOV multipliers (sprinting, speed effects, etc.)
    // - Underwater/lava FOV changes
    return 70.0f;
}

void GameRenderer::renderWorld() {
    viewDistanceBlocks = static_cast<float>(gameOptions.getClampedViewDistance() * 16);
}
