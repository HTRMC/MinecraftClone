#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in flat uint fragTextureIndex;

layout(location = 0) out vec4 outColor;

void main() {
    // Output the vertex color from the mesh shader
    // The mesh shader already applies lighting calculations
    outColor = vec4(fragColor, 1.0);
}