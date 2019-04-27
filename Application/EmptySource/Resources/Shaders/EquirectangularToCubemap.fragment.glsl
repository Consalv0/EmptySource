#version 410 core

const float PI = 3.14159265359;
const vec2 InvAtan = vec2(0.1591, 0.3183);

uniform sampler2D _EquirectangularMap;
uniform float _Roughness;

in vec3 LocalPosition;

out vec4 FragColor;

float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N) {
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float Roughness) {
    float RoughnessSqr = Roughness * Roughness;
	
    float Phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (RoughnessSqr*RoughnessSqr - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    // --- From spherical coordinates to cartesian coordinates
    vec3 H;
    H.x = cos(Phi) * sinTheta;
    H.y = sin(Phi) * sinTheta;
    H.z = cosTheta;
	
    // --- From tangent-space vector to world-space sample vector
    vec3 Up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 Tangent   = normalize(cross(Up, N));
    vec3 Bitangent = cross(N, Tangent);
	
    return (Tangent * H.x + Bitangent * H.y + N * H.z);
} 

vec2 SampleSphericalMap(vec3 Vector) {
    vec2 UV = vec2(atan(Vector.z, Vector.x), asin(Vector.y));
    UV *= InvAtan;
    UV += 0.5;
    return UV;
}

float DistributionGGX(vec3 N, vec3 H, float Roughness) {
    float a      = Roughness*Roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

void main() {
    vec3 N = normalize(LocalPosition);    
    vec3 R = N;
    vec3 V = R;

    const uint SAMPLE_COUNT = 256u;
    float TotalWeight = 0.0;   
    vec3 PrefilteredColor = vec3(0.0);     
    vec3 ExccessColor = vec3(0.0);
    for(uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H  = ImportanceSampleGGX(Xi, N, _Roughness);
        vec3 L  = 2.0 * dot(V, H) * H - V;

        float NdotL = max(dot(N, L), 0.0);
        if(NdotL > 0.0) {
            vec2 UV = SampleSphericalMap(normalize(L));
            float D = DistributionGGX(N, H, _Roughness);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);
            float PDF = (D * NdotH / (4.0 * HdotV)) + 0.0001;

            float TexelSize  = 4.0 * PI / (6.0 * 2048 * 1024);
            float SampleSize = 1.0 / (float(SAMPLE_COUNT) * PDF + 0.0001);

            float MipLevel = _Roughness == 0.0 ? 0.0 : 0.5 * log2(SampleSize / TexelSize); 

            vec3 Color = textureLod(_EquirectangularMap, UV, MipLevel).rgb;
            PrefilteredColor += Color * NdotL;
            TotalWeight      += NdotL;
        }
    }
    PrefilteredColor = PrefilteredColor / max(TotalWeight, 0.001);
    vec3 Intensity = vec3(dot(PrefilteredColor, vec3(0.2125, 0.7154, 0.0721)));
    PrefilteredColor = mix(Intensity, PrefilteredColor, 1 + _Roughness * 0.3);

    FragColor = vec4(PrefilteredColor, 1.0);
}