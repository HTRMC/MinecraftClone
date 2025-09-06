#include "BlockModelLoader.hpp"
#include "Logger.hpp"
#include <fstream>
#include <stdexcept>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

BlockModel BlockModelLoader::loadModelWithInheritance(const std::string& modelPath) {
    BlockModel model = loadModel(modelPath);
    
    // If this model has a parent, load and merge it
    if (model.parent.has_value()) {
        std::string parentPath = resolveModelPath(model.parent.value());
        try {
            BlockModel parentModel = loadModelWithInheritance(parentPath); // Recursive call for nested inheritance
            model = mergeWithParent(model, parentModel);
        } catch (const std::exception& e) {
            Logger::warning("BlockModelLoader", "Failed to load parent model '" + model.parent.value() + "': " + std::string(e.what()));
        }
    }
    
    return model;
}

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

    // Parse parent
    auto parentValue = doc["parent"];
    if (!parentValue.error()) {
        std::string_view parent;
        if (parentValue.get(parent) == simdjson::SUCCESS) {
            model.parent = std::string(parent);
        }
    }

    // Parse ambient occlusion
    auto aoValue = doc["ambientocclusion"];
    if (!aoValue.error()) {
        bool ao;
        if (aoValue.get(ao) == simdjson::SUCCESS) {
            model.ambientocclusion = ao;
        }
    }

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
            
            // Parse rotation (optional)
            auto rotationObj = elementValue["rotation"];
            if (!rotationObj.error()) {
                element.rotation = parseRotation(simdjson::dom::object(rotationObj));
            }
            
            // Parse shade (optional, default true)
            auto shadeValue = elementValue["shade"];
            if (!shadeValue.error()) {
                bool shade;
                if (shadeValue.get(shade) == simdjson::SUCCESS) {
                    element.shade = shade;
                }
            }
            
            // Parse light (optional, default 0)
            auto lightValue = elementValue["light"];
            if (!lightValue.error()) {
                uint64_t light;
                if (lightValue.get(light) == simdjson::SUCCESS) {
                    element.light = static_cast<int>(light);
                }
            }
            
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
            auto faceIt = element.faces.find(faceName);
            if (faceIt != element.faces.end()) {
                const BlockFace& face = faceIt->second;
                ModelData modelData = {};
                
                // Generate vertices for this face
                std::vector<glm::vec4> vertices = generateFaceVertices(from, to, faceName, element.rotation);
                for (int i = 0; i < 4; i++) {
                    modelData.vertices[i] = vertices[i];
                }
                
                // Generate UV coordinates from face data
                FaceUV uv = face.uv;
                glm::vec2 uvCoords[4] = {
                    glm::vec2(uv.minU / 16.0f, uv.minV / 16.0f),
                    glm::vec2(uv.maxU / 16.0f, uv.minV / 16.0f),
                    glm::vec2(uv.maxU / 16.0f, uv.maxV / 16.0f),
                    glm::vec2(uv.minU / 16.0f, uv.maxV / 16.0f)
                };
                
                // Apply texture rotation
                for (int i = 0; i < 4; i++) {
                    modelData.uvCoords[i] = rotateUV(uvCoords[i].x, uvCoords[i].y, face.rotation);
                }
                
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

std::unordered_map<std::string, BlockFace> BlockModelLoader::parseFaces(simdjson::dom::object facesObj) {
    std::unordered_map<std::string, BlockFace> faces;
    for (auto [key, value] : facesObj) {
        auto faceObj = simdjson::dom::object(value);
        faces[std::string(key)] = parseFace(faceObj);
    }
    return faces;
}

BlockFace BlockModelLoader::parseFace(simdjson::dom::object faceObj) {
    BlockFace face;
    
    // Parse texture (required)
    auto textureValue = faceObj["texture"];
    if (!textureValue.error()) {
        std::string_view texture;
        if (textureValue.get(texture) == simdjson::SUCCESS) {
            face.texture = std::string(texture);
        }
    }
    
    // Parse cullface (optional)
    auto cullfaceValue = faceObj["cullface"];
    if (!cullfaceValue.error()) {
        std::string_view cullface;
        if (cullfaceValue.get(cullface) == simdjson::SUCCESS) {
            face.cullface = std::string(cullface);
        }
    }
    
    // Parse tintindex (optional, default -1)
    auto tintValue = faceObj["tintindex"];
    if (!tintValue.error()) {
        uint64_t tint;
        if (tintValue.get(tint) == simdjson::SUCCESS) {
            face.tintindex = static_cast<int>(tint);
        }
    }
    
    // Parse uv (optional)
    auto uvArray = faceObj["uv"];
    if (!uvArray.error()) {
        face.uv = parseUV(simdjson::dom::array(uvArray));
    }
    
    // Parse rotation (optional, default 0)
    auto rotationValue = faceObj["rotation"];
    if (!rotationValue.error()) {
        uint64_t rotation;
        if (rotationValue.get(rotation) == simdjson::SUCCESS) {
            face.rotation = static_cast<int>(rotation);
        }
    }
    
    return face;
}

FaceUV BlockModelLoader::parseUV(simdjson::dom::array uvArray) {
    auto it = uvArray.begin();
    float minU = double(*it++);
    float minV = double(*it++);
    float maxU = double(*it++);
    float maxV = double(*it++);
    return FaceUV(minU, minV, maxU, maxV);
}

std::optional<ElementRotation> BlockModelLoader::parseRotation(simdjson::dom::object rotationObj) {
    ElementRotation rotation;
    
    // Parse origin
    auto originArray = rotationObj["origin"];
    if (originArray.error()) return std::nullopt;
    rotation.origin = parseVector3(simdjson::dom::array(originArray));
    
    // Parse axis
    auto axisValue = rotationObj["axis"];
    if (axisValue.error()) return std::nullopt;
    std::string_view axis;
    if (axisValue.get(axis) != simdjson::SUCCESS) return std::nullopt;
    
    if (axis == "x") rotation.axis = RotationAxis::X;
    else if (axis == "y") rotation.axis = RotationAxis::Y;
    else if (axis == "z") rotation.axis = RotationAxis::Z;
    else return std::nullopt;
    
    // Parse angle
    auto angleValue = rotationObj["angle"];
    if (angleValue.error()) return std::nullopt;
    double angle;
    if (angleValue.get(angle) != simdjson::SUCCESS) return std::nullopt;
    rotation.angle = static_cast<float>(angle);
    
    // Parse rescale (optional, default false)
    auto rescaleValue = rotationObj["rescale"];
    if (!rescaleValue.error()) {
        bool rescale;
        if (rescaleValue.get(rescale) == simdjson::SUCCESS) {
            rotation.rescale = rescale;
        }
    } else {
        rotation.rescale = false;
    }
    
    return rotation;
}

glm::mat4 BlockModelLoader::createRotationMatrix(const ElementRotation& rotation) {
    glm::mat4 transform = glm::mat4(1.0f);
    
    // Convert origin from [0-16] space to [-0.5, 0.5] space
    glm::vec3 origin = (rotation.origin / 16.0f) - 0.5f;
    
    // Translate to origin
    transform = glm::translate(transform, origin);
    
    // Apply rotation
    float radians = glm::radians(rotation.angle);
    switch (rotation.axis) {
        case RotationAxis::X:
            transform = glm::rotate(transform, radians, glm::vec3(1, 0, 0));
            break;
        case RotationAxis::Y:
            transform = glm::rotate(transform, radians, glm::vec3(0, 1, 0));
            break;
        case RotationAxis::Z:
            transform = glm::rotate(transform, radians, glm::vec3(0, 0, 1));
            break;
    }
    
    // Translate back
    transform = glm::translate(transform, -origin);
    
    return transform;
}

std::vector<glm::vec4> BlockModelLoader::generateFaceVertices(const glm::vec3& from, const glm::vec3& to, const std::string& face, const std::optional<ElementRotation>& rotation) {
    std::vector<glm::vec4> vertices(4);
    
    // Generate base vertices
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
    
    // Apply rotation if specified
    if (rotation.has_value()) {
        glm::mat4 rotationMatrix = createRotationMatrix(rotation.value());
        for (auto& vertex : vertices) {
            vertex = rotationMatrix * vertex;
        }
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

glm::vec2 BlockModelLoader::rotateUV(float u, float v, int rotation) {
    switch (rotation) {
        case 0:   return glm::vec2(u, v);
        case 90:  return glm::vec2(1.0f - v, u);
        case 180: return glm::vec2(1.0f - u, 1.0f - v);
        case 270: return glm::vec2(v, 1.0f - u);
        default:  return glm::vec2(u, v);
    }
}

std::string BlockModelLoader::resolveModelPath(const std::string& modelId) {
    // Handle Minecraft-style model IDs like "minecraft:block/cube"
    if (modelId.find(':') != std::string::npos) {
        // Convert "minecraft:block/cube" to "assets/minecraft/models/block/cube.json"
        size_t colonPos = modelId.find(':');
        std::string namespace_ = modelId.substr(0, colonPos);
        std::string path = modelId.substr(colonPos + 1);
        
        return "assets/" + namespace_ + "/models/" + path + ".json";
    } else {
        // Handle relative paths like "block/cube" - assume minecraft namespace
        if (modelId.find('/') != std::string::npos) {
            return "assets/minecraft/models/" + modelId + ".json";
        } else {
            // Assume it's already a file path
            return modelId;
        }
    }
}

BlockModel BlockModelLoader::mergeWithParent(const BlockModel& child, const BlockModel& parent) {
    BlockModel merged = child;
    
    // Inherit ambient occlusion if not specified in child
    if (!merged.ambientocclusion.has_value() && parent.ambientocclusion.has_value()) {
        merged.ambientocclusion = parent.ambientocclusion;
    }
    
    // Merge textures - child textures override parent textures
    for (const auto& [key, value] : parent.textures) {
        if (merged.textures.find(key) == merged.textures.end()) {
            merged.textures[key] = value;
        }
    }
    
    // Resolve texture references (e.g., "#particle" -> actual texture)
    std::unordered_map<std::string, std::string> resolvedTextures;
    for (const auto& [key, value] : merged.textures) {
        resolvedTextures[key] = resolveTextureReference(value, merged.textures);
    }
    merged.textures = resolvedTextures;
    
    // If child has no elements, inherit parent's elements
    if (merged.elements.empty() && !parent.elements.empty()) {
        merged.elements = parent.elements;
    }
    
    // Clear parent reference as it's been resolved
    merged.parent.reset();
    
    Logger::debug("BlockModelLoader", "Merged model with parent - textures: " + std::to_string(merged.textures.size()) + 
                  ", elements: " + std::to_string(merged.elements.size()));
    
    return merged;
}

std::string BlockModelLoader::resolveTextureReference(const std::string& textureRef, const std::unordered_map<std::string, std::string>& textures) {
    if (textureRef.empty() || textureRef[0] != '#') {
        return textureRef; // Not a reference, return as-is
    }
    
    // Remove the '#' prefix
    std::string key = textureRef.substr(1);
    
    // Look up the reference
    auto it = textures.find(key);
    if (it != textures.end()) {
        // Recursively resolve in case the referenced texture is also a reference
        return resolveTextureReference(it->second, textures);
    }
    
    Logger::warning("BlockModelLoader", "Unresolved texture reference: " + textureRef);
    return textureRef; // Return original if not found
}