#pragma once
#include "client/vulkan/MeshShaderData.hpp"
#include <simdjson.h>
#include <string>
#include <vector>
#include <unordered_map>

struct BlockElement {
    glm::vec3 from;
    glm::vec3 to;
    std::unordered_map<std::string, std::string> faces;
};

struct BlockModel {
    std::unordered_map<std::string, std::string> textures;
    std::vector<BlockElement> elements;
};

class BlockModelLoader {
public:
    static BlockModel loadModel(const std::string& modelPath);
    static std::vector<ModelData> generateMeshData(const BlockModel& model);

private:
    static glm::vec3 parseVector3(simdjson::dom::array array);
    static std::unordered_map<std::string, std::string> parseFaces(simdjson::dom::object facesObj);
    static std::vector<glm::vec4> generateFaceVertices(const glm::vec3& from, const glm::vec3& to, const std::string& face);
    static glm::vec3 getFaceNormal(const std::string& face);
};