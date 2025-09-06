#include "BlockModelLoader.hpp"
#include "Logger.hpp"
#include <fstream>
#include <stdexcept>

BlockModel BlockModelLoader::loadModel(const std::string& modelPath) {
    std::ifstream file(modelPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open model file: " + modelPath);
    }

    std::string jsonContent((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    file.close();

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    auto error = parser.parse(jsonContent).get(doc);
    if (error) {
        throw std::runtime_error("Failed to parse JSON: " + std::string(simdjson::error_message(error)));
    }

    BlockModel model;

    // Parse textures
    auto texturesObj = doc["textures"];
    if (!texturesObj.error()) {
        for (auto [key, value] : simdjson::dom::object(texturesObj)) {
            std::string_view textureValue;
            if (value.get(textureValue) == simdjson::SUCCESS) {
                model.textures[std::string(key)] = std::string(textureValue);
            }
        }
    }

    // Parse elements
    auto elementsArray = doc["elements"];
    if (!elementsArray.error()) {
        for (auto elementValue : simdjson::dom::array(elementsArray)) {
            BlockElement element;
            
            // Parse 'from' vector
            auto fromArray = elementValue["from"];
            element.from = parseVector3(simdjson::dom::array(fromArray));
            
            // Parse 'to' vector  
            auto toArray = elementValue["to"];
            element.to = parseVector3(simdjson::dom::array(toArray));
            
            // Parse faces
            auto facesObj = elementValue["faces"];
            if (!facesObj.error()) {
                element.faces = parseFaces(simdjson::dom::object(facesObj));
            }
            
            model.elements.push_back(element);
        }
    }

    Logger::info("BlockModelLoader", "Loaded model with " + std::to_string(model.elements.size()) + " elements");
    return model;
}

std::vector<ModelData> BlockModelLoader::generateMeshData(const BlockModel& model) {
    std::vector<ModelData> meshData;

    for (const auto& element : model.elements) {
        // Convert from [0-16] space to [-0.5, 0.5] space
        glm::vec3 from = (element.from / 16.0f) - 0.5f;
        glm::vec3 to = (element.to / 16.0f) - 0.5f;

        // Generate faces in the same order as the original hardcoded version
        std::vector<std::string> faceOrder = {"west", "east", "north", "south", "down", "up"};
        
        for (const std::string& faceName : faceOrder) {
            if (element.faces.find(faceName) != element.faces.end()) {
                ModelData modelData = {};
                
                // Generate vertices for this face
                std::vector<glm::vec4> vertices = generateFaceVertices(from, to, faceName);
                for (int i = 0; i < 4; i++) {
                    modelData.vertices[i] = vertices[i];
                }
                
                // Standard UV coordinates
                modelData.uvCoords[0] = glm::vec2(0.0f, 0.0f);
                modelData.uvCoords[1] = glm::vec2(1.0f, 0.0f);
                modelData.uvCoords[2] = glm::vec2(1.0f, 1.0f);
                modelData.uvCoords[3] = glm::vec2(0.0f, 1.0f);
                
                // Set face normal
                glm::vec3 normal = getFaceNormal(faceName);
                modelData.faceNormal = glm::vec4(normal, 0.0f);
                
                meshData.push_back(modelData);
            }
        }
    }

    Logger::info("BlockModelLoader", "Generated " + std::to_string(meshData.size()) + " faces");
    return meshData;
}

glm::vec3 BlockModelLoader::parseVector3(simdjson::dom::array array) {
    auto it = array.begin();
    float x = double(*it++);
    float y = double(*it++);
    float z = double(*it++);
    return glm::vec3(x, y, z);
}

std::unordered_map<std::string, std::string> BlockModelLoader::parseFaces(simdjson::dom::object facesObj) {
    std::unordered_map<std::string, std::string> faces;
    for (auto [key, value] : facesObj) {
        auto faceObj = simdjson::dom::object(value);
        auto textureValue = faceObj["texture"];
        if (!textureValue.error()) {
            std::string_view texture;
            if (textureValue.get(texture) == simdjson::SUCCESS) {
                faces[std::string(key)] = std::string(texture);
            }
        }
    }
    return faces;
}

std::vector<glm::vec4> BlockModelLoader::generateFaceVertices(const glm::vec3& from, const glm::vec3& to, const std::string& face) {
    std::vector<glm::vec4> vertices(4);
    
    if (face == "west") {
        // Left face (-X) - Counter-clockwise when viewed from outside
        vertices[0] = glm::vec4(from.x, from.y, from.z, 1.0f);
        vertices[1] = glm::vec4(from.x, to.y, from.z, 1.0f);
        vertices[2] = glm::vec4(from.x, to.y, to.z, 1.0f);
        vertices[3] = glm::vec4(from.x, from.y, to.z, 1.0f);
    } else if (face == "east") {
        // Right face (+X) - Counter-clockwise when viewed from outside
        vertices[0] = glm::vec4(to.x, from.y, to.z, 1.0f);
        vertices[1] = glm::vec4(to.x, to.y, to.z, 1.0f);
        vertices[2] = glm::vec4(to.x, to.y, from.z, 1.0f);
        vertices[3] = glm::vec4(to.x, from.y, from.z, 1.0f);
    } else if (face == "north") {
        // Front face (-Y) - Counter-clockwise when viewed from outside
        vertices[0] = glm::vec4(to.x, from.y, from.z, 1.0f);
        vertices[1] = glm::vec4(from.x, from.y, from.z, 1.0f);
        vertices[2] = glm::vec4(from.x, from.y, to.z, 1.0f);
        vertices[3] = glm::vec4(to.x, from.y, to.z, 1.0f);
    } else if (face == "south") {
        // Back face (+Y) - Counter-clockwise when viewed from outside
        vertices[0] = glm::vec4(from.x, to.y, from.z, 1.0f);
        vertices[1] = glm::vec4(to.x, to.y, from.z, 1.0f);
        vertices[2] = glm::vec4(to.x, to.y, to.z, 1.0f);
        vertices[3] = glm::vec4(from.x, to.y, to.z, 1.0f);
    } else if (face == "down") {
        // Bottom face (-Z) - Counter-clockwise when viewed from outside
        vertices[0] = glm::vec4(from.x, from.y, from.z, 1.0f);
        vertices[1] = glm::vec4(to.x, from.y, from.z, 1.0f);
        vertices[2] = glm::vec4(to.x, to.y, from.z, 1.0f);
        vertices[3] = glm::vec4(from.x, to.y, from.z, 1.0f);
    } else if (face == "up") {
        // Top face (+Z) - Counter-clockwise when viewed from outside
        vertices[0] = glm::vec4(from.x, from.y, to.z, 1.0f);
        vertices[1] = glm::vec4(from.x, to.y, to.z, 1.0f);
        vertices[2] = glm::vec4(to.x, to.y, to.z, 1.0f);
        vertices[3] = glm::vec4(to.x, from.y, to.z, 1.0f);
    }
    
    return vertices;
}

glm::vec3 BlockModelLoader::getFaceNormal(const std::string& face) {
    if (face == "west") return glm::vec3(-1, 0, 0);
    if (face == "east") return glm::vec3(1, 0, 0);
    if (face == "north") return glm::vec3(0, -1, 0);
    if (face == "south") return glm::vec3(0, 1, 0);
    if (face == "down") return glm::vec3(0, 0, -1);
    if (face == "up") return glm::vec3(0, 0, 1);
    return glm::vec3(0, 0, 0);
}