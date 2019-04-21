#version 410

const float PI = 3.1415926535;
const float Gamma = 2.2;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform vec3 _ViewPosition;

uniform sampler2D _MainTexture;
uniform sampler2D _RoughnessTexture;
uniform sampler2D _MetallicTexture;
uniform float _EnviromentMapLods;
uniform samplerCube _EnviromentMap;
uniform sampler2D   _BRDFLUT;

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

in struct Matrices {
  mat4 Model;
  mat4 WorldNormal;
} Matrix;

in struct VertexData {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec2 UV0;
  vec4 Color;
} Vertex;

out vec4 FragColor;

vec3 FresnelSchlick(float CosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - CosTheta, 5.0);
}

vec3 FresnelSchlickRoughness(float CosTheta, vec3 F0, float Roughness) {
    return F0 + (max(vec3(1.0 - Roughness), F0) - F0) * pow(1.0 - CosTheta, 5.0);
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

// Enviroment Light
vec3 MicrofacetModelEnviroment( vec3 VertPosition, vec3 VertNormal ) {

  vec3 Normal = normalize(VertNormal);

  vec3 EyeDirection = normalize(_ViewPosition.xyz - VertPosition.xyz);
  vec3 WorldReflection = reflect(-EyeDirection, normalize(Normal));
  
  float Metalness = _Material.Metalness * texture(_MetallicTexture, Vertex.UV0).r;
  float Roughness = _Material.Roughness * texture(_RoughnessTexture, Vertex.UV0).r;
  vec3 DiffuseColor = pow(texture(_MainTexture, Vertex.UV0).rgb, vec3(Gamma));
  vec3 SpecularColor = mix(vec3(1), _Material.Color, Metalness);
  
  vec3 F0 = vec3(SpecularColor * 0.04); 
  F0 = mix(F0, DiffuseColor, Metalness);
  vec3 F = FresnelSchlickRoughness(max(dot(Normal, EyeDirection), 0.0), F0, Roughness);
  vec2 EnviromentBRDF  = texture(_BRDFLUT, vec2(max(dot(Normal, EyeDirection), 0.0), Roughness)).rg;
  vec3 EnviromentLight = vec3(textureLod(_EnviromentMap, WorldReflection, Roughness * (_EnviromentMapLods - 4)));

  vec3 Specular = EnviromentLight * (F * EnviromentBRDF.x + EnviromentBRDF.y);

  EnviromentLight = vec3(textureLod(_EnviromentMap, WorldReflection, _EnviromentMapLods * 0.5));

  // vec3 Color = (kD * DiffuseColor * EnviromentLight) + Specular;
  vec3 Color = (1 - (F * Metalness)) * DiffuseColor * EnviromentLight + Specular;
  return Color;
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
  float HDotV = clamp( dot( HalfWayDirection, EyeDirection ), 0, 1 );

  float Roughness = _Material.Roughness * texture(_RoughnessTexture, Vertex.UV0).r;
  float Metalness = _Material.Metalness * texture(_MetallicTexture, Vertex.UV0).r;
  float InvSqrLightRadius = 1 / ( (_Lights[LightIndex].Intencity * _Lights[LightIndex].Intencity) + 0.001) ;
  float Attenuation = GetLightAttenuation(UnormalizedLightDirection, InvSqrLightRadius) * _Lights[LightIndex].Intencity;
  
  vec3 NormalColor = (Matrix.WorldNormal * vec4(Normal, 1)).rgb;
       NormalColor = normalize(NormalColor) * 0.5 + 0.5;

  vec3 SpecularColor = mix(vec3(1), _Material.Color, Metalness);
  vec3 DiffuseColor = pow(texture(_MainTexture, Vertex.UV0).rgb, vec3(Gamma));
  vec3 F0 = vec3(SpecularColor * 0.04); 
  F0 = mix(F0, DiffuseColor, Metalness);
  
  float SpecularDistribution = TrowbridgeReitzNormalDistribution(NDotH, Roughness);
  float GeometricShadow = WalterEtAlGeometricShadowingFunction(NDotL, NDotV, Roughness);
  vec3  Fresnel = FresnelSchlick(HDotV, F0);

  vec3 Diffuse = DiffuseColor * _Lights[LightIndex].Color;
  vec3 Specular = (SpecularDistribution * Fresnel * GeometricShadow) / (4 * (NDotL * NDotV));
  
  vec3 SurfaceColor = (Diffuse + Specular) * Attenuation * NDotL;

  return max(SurfaceColor, 0);
}

void main() {  
  vec3 Sum = vec3(0);

  for( int i = 0; i < 2; i++ ) {
    Sum += MicrofacetModel(i, Vertex.Position.xyz, Vertex.NormalDirection);
  }

  // Sum += MicrofacetModelEnviroment(Vertex.Position.xyz, Vertex.NormalDirection);
  Sum = Sum / (Sum + vec3(1.0));
  Sum = pow(Sum, vec3(1.0/Gamma));

  FragColor = vec4(Sum, 1);
}
