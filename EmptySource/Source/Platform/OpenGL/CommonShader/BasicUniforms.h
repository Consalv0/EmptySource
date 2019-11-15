const ESource::NString CommonUniforms = R"(
uniform mat4 _ModelMatrix;
uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform vec3 _ViewPosition;
)";

const ESource::NString CommonUniformMaterial = R"(
uniform sampler2D   _MainTexture;
uniform sampler2D   _NormalTexture;
uniform sampler2D   _RoughnessTexture;
uniform sampler2D   _MetallicTexture;
uniform sampler2D   _AOTexture;
uniform sampler2D   _EmissionTexture;
uniform vec3        _EmissionColor;
        
uniform struct MaterialInfo {
    float Roughness;
    float Metalness;
    vec4 Color;
} _Material;
)";