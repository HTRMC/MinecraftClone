#version 450

layout(binding = 1) uniform sampler2DArray texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in flat uint fragBlockType;

layout(location = 0) out vec4 outColor;

void main() {
    // Sample from texture array using block type as layer index
    vec4 texColor = texture(texSampler, vec3(fragTexCoord, float(fragBlockType)));

    // Mix with face color for better visibility
    outColor = texColor * vec4(fragColor, 1.0);
}