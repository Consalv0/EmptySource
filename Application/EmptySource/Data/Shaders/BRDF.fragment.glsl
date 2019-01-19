#version 460

const float PI = 3.1415926535;
const float Gamma = 2.2;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform vec3 _ViewPosition;

uniform struct LightInfo {
	vec3 Position;        // Light Position in camera coords.
  vec3 Color;           // Light Color
	float Intencity;      // Light Intensity
} _Lights[2];

uniform struct MaterialInfo {
  float Roughness;     // Roughness
  float Metalness;     // Metallic (0) or dielectric (1)
  vec3 Color;          // Diffuse color for dielectrics, f0 for metallic
} _Material;

in mat4 ModelMatrix;
in mat4 WorldNormalMatrix;

in vec4 VertexPosition;
in vec3 NormalDirection;
in vec3 TangentDirection;
in vec2 UV0;
in vec4 Color;

out vec4 FragColor;

float SchlickFresnel(float i) {
    float x = clamp(1.0 - i, 0.0, 1.0);
    float x2 = x * x;
    return x2 * x2 * x;
}

float FresnelIncidenceReflection(float NDotL, float NDotV, float LDotH, float Roughness) {
    float FresnelLight = SchlickFresnel(NDotL); 
    float FresnelView = SchlickFresnel(NDotV);
    float FresnelDiffuse90 = 0.5 + 2.0 * LDotH * LDotH * Roughness;
    return mix(1, FresnelDiffuse90, FresnelLight) * mix(1, FresnelDiffuse90, FresnelView);
}

float WalterEtAlGeometricShadowingFunction (float NDotL, float NDotV, float RoughnessSqr){
    float NDotLSqr = NDotL*NDotL;
    float NDotVSqr = NDotV*NDotV;

    float SmithLight = 2/(1 + sqrt(1 + RoughnessSqr * (1-NDotLSqr)/(NDotLSqr)));
    float SmithVisibility = 2/(1 + sqrt(1 + RoughnessSqr * (1-NDotVSqr)/(NDotVSqr)));

	float GeometricShadow = (SmithLight * SmithVisibility);
	return GeometricShadow;
}

float TrowbridgeReitzNormalDistribution(float NdotH, float RoughnessSqr){
    float Distribution = NdotH*NdotH * (RoughnessSqr-1.0) + 1.0;
    return RoughnessSqr / (PI * Distribution*Distribution);
}

float SmoothDistanceAttenuation (float SquaredDistance, float InvSqrAttenuationRadius) {
  float Factor = SquaredDistance * InvSqrAttenuationRadius;
  float SmoothFactor = clamp(1.0f - Factor * Factor, 0, 1);
  return SmoothFactor * SmoothFactor;
}

float GetLightAttenuation (vec3 UnormalizedLightVector, float InvSqrAttenuationRadius) {
  float SquaredDistance = dot(UnormalizedLightVector, UnormalizedLightVector );
  float Attenuation = 1.0 / ( max(SquaredDistance, 0.01*0.01) );
  Attenuation *= SmoothDistanceAttenuation(SquaredDistance, InvSqrAttenuationRadius);
  return Attenuation;
}

// Light Calculation
vec3 MicrofacetModel( int LightIndex, vec3 VertPosition, vec3 VertNormal ) {

  vec3 Normal = normalize(VertNormal);

  vec3 UnormalizedLightDirection = (_Lights[LightIndex].Position - VertPosition).xyz;
  float LightDistance = length(UnormalizedLightDirection);

  vec3 LightDirection = normalize(UnormalizedLightDirection);
  vec3 EyeDirection = normalize(_ViewPosition.xyz - VertPosition.xyz);
  vec3 HalfWayDirection = normalize(EyeDirection + LightDirection);
  
  float LDotH = clamp( dot( LightDirection, HalfWayDirection ), 0, 1 );
  float NDotH = clamp( dot( Normal, HalfWayDirection ), 0, 1 );
  float NDotL = clamp( dot( Normal, LightDirection ), 0, 1 );
  float NDotV = abs( dot (Normal , EyeDirection )) + 1e-5f; // Avoids artifact

  float InvSqrLightRadius = 1 / ( (_Lights[LightIndex].Intencity * _Lights[LightIndex].Intencity) + 1e-5f) ;
  float Attenuation = 1;
        // Attenuation *= ShadowCalculation(positionLightSpace, 0.0001);
        // Attenuation *= (-dot(LightDirection * 2, LightDirection) - 1.4) * 6;
        Attenuation *= GetLightAttenuation(UnormalizedLightDirection, InvSqrLightRadius) * _Lights[LightIndex].Intencity;
  
  vec3 NormalColor = (_ViewMatrix * vec4(Normal, 1)).rgb;
       NormalColor = (1 - NormalColor) * 0.5;

  float LinearRoughness = _Material.Roughness;
  float Roughness = LinearRoughness * LinearRoughness;
  float Metalness = _Material.Metalness;

  vec3 SpecularColor = mix(vec3(1), _Material.Color, Metalness);
  vec3 DiffuseColor = NormalColor * (1 - Metalness);
  // DiffuseColor *= texture(_texture, uv).rgb;
  
  float SpecularDistribution = TrowbridgeReitzNormalDistribution(NDotH, Roughness);
  float GeometricShadow = WalterEtAlGeometricShadowingFunction(NDotL, NDotV, Roughness);
  float Fresnel = FresnelIncidenceReflection(NDotL, NDotV, NDotH, Roughness);

  vec3 Diffuse = NDotL * DiffuseColor * _Lights[LightIndex].Color;
  vec3 Specular = SpecularColor;
       Specular *= (SpecularDistribution * Fresnel * GeometricShadow) / (4 * (  NDotL * NDotV));
  
  vec3 SurfaceColor = (Diffuse + Specular) * Attenuation * NDotL;

  return max (SurfaceColor, 0);
}

void main() {  
  vec3 Sum = vec3(0);

  for( int i = 0; i < 2; i++ ) {
    Sum += MicrofacetModel(i, VertexPosition.xyz, NormalDirection);
  }
 
  Sum = pow( Sum, vec3(1.0/Gamma) );

  FragColor = vec4(Sum, 1);
}