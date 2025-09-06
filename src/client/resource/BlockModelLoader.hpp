#pragma once
#include "client/vulkan/MeshShaderData.hpp"
#include <simdjson.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

enum class RotationAxis {
    X, Y, Z
};

struct ElementRotation {
    glm::vec3 origin;
    RotationAxis axis;
    float angle;
    bool rescale;
};

struct FaceUV {
    float minU, minV, maxU, maxV;
    
    FaceUV() : minU(0), minV(0), maxU(16), maxV(16) {}
    FaceUV(float minU, float minV, float maxU, float maxV) 
        : minU(minU), minV(minV), maxU(maxU), maxV(maxV) {}
};

struct BlockFace {
    std::string texture;
    std::optional<std::string> cullface;
    int tintindex = -1;
    FaceUV uv;
    int rotation = 0; // 0, 90, 180, 270 degrees
};

struct BlockElement {
    glm::vec3 from;
    glm::vec3 to;
    std::unordered_map<std::string, BlockFace> faces;
    std::optional<ElementRotation> rotation;
    bool shade = true;
    int light = 0;
};

struct BlockModel {
    std::unordered_map<std::string, std::string> textures;
    std::vector<BlockElement> elements;
    std::optional<std::string> parent;
    std::optional<bool> ambientocclusion;
};

class BlockModelLoader {
public:
    static BlockModel loadModel(const std::string& modelPath);
    static std::vector<ModelData> generateMeshData(const BlockModel& model);

private:
    static glm::vec3 parseVector3(simdjson::dom::array array);
    static std::unordered_map<std::string, BlockFace> parseFaces(simdjson::dom::object facesObj);
    static BlockFace parseFace(simdjson::dom::object faceObj);
    static FaceUV parseUV(simdjson::dom::array uvArray);
    static std::optional<ElementRotation> parseRotation(simdjson::dom::object rotationObj);
    static std::vector<glm::vec4> generateFaceVertices(const glm::vec3& from, const glm::vec3& to, const std::string& face, const std::optional<ElementRotation>& rotation = std::nullopt);
    static glm::vec3 getFaceNormal(const std::string& face);
    static glm::vec2 rotateUV(float u, float v, int rotation);
    static glm::mat4 createRotationMatrix(const ElementRotation& rotation);
};