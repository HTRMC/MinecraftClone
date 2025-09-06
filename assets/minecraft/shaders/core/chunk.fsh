#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in flat uint fragTextureIndex;

layout(binding = 5) uniform texture2D textures[];
layout(binding = 6) uniform sampler texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    // Sample texture using bindless array
    vec4 textureColor = texture(sampler2D(textures[nonuniformEXT(fragTextureIndex)], texSampler), fragUV);
    
    // Combine texture color with lighting
    outColor = vec4(fragColor * textureColor.rgb, textureColor.a);
}