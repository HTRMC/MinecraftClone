#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple textured output with lighting
    // Note: Texture sampling would require texture binding
    // For now, output the vertex color with basic lighting
    
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    float NdotL = max(dot(normalize(fragNormal), lightDir), 0.1);
    
    vec3 baseColor = fragColor.rgb;
    vec3 finalColor = baseColor * NdotL;
    
    outColor = vec4(finalColor, fragColor.a);
}