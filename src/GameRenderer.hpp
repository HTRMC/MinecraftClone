#pragma once

#include "GameOptions.hpp"

class GameRenderer {

private:
    float viewDistanceBlocks;
    GameOptions gameOptions;

public:
    float getFarPlaneDistance();
    float getFov();
    void renderWorld();
};
