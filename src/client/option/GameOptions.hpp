#pragma once
#include "SimpleOption.hpp"

class GameOptions {
private:
    SimpleOption<int> viewDistance;
    int serverViewDistance;
    SimpleOption<int> cloudRenderDistance;

public:
    GameOptions();
    int getClampedViewDistance();
    SimpleOption<int>& getCloudRenderDistance();
};