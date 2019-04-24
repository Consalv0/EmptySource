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

float GeometrySmithSchlickGGX(float NDotL, float NDotV, float Roughness) {
  float RoughnessPlusOne = (Roughness);
  float k = (RoughnessPlusOne * RoughnessPlusOne) / 8.0;

  float GGXDotV = NDotV / ( NDotV * (1.0 - k) + k );
  float GGXDotL = NDotL / ( NDotL * (1.0 - k) + k );

  return GGXDotV * GGXDotL;
}

float TrowbridgeReitzNormalDistribution(float NdotH, float Roughness){
  float RoughnessSqr = Roughness * Roughness;
  float Distribution = NdotH*NdotH * (RoughnessSqr - 1.0) + 1.0;
  return RoughnessSqr / (PI * Distribution*Distribution);
}

float SmoothDistanceAttenuation (vec3 UnormalizedLightVector, float AttenuationRadius) {
  float SquaredDistance = dot(UnormalizedLightVector, UnormalizedLightVector) * (1 / AttenuationRadius);

  float Attenuation = 1.0 / ( max(SquaredDistance, 0.001) );
  return Attenuation;
}

// Enviroment Light
vec3 MicrofacetModelEnviroment( vec3 VertPosition, vec3 VertNormal ) {

  vec3 Normal = normalize(VertNormal);

  vec3 EyeDirection = normalize(_ViewPosition.xyz - VertPosition.xyz);
  vec3 WorldReflection = reflect(-EyeDirection, normalize(Normal));
  
  float Metalness = _Material.Metalness * texture(_MetallicTexture, Vertex.UV0).r;
  float Roughness = _Material.Roughness * texture(_RoughnessTexture, Vertex.UV0).r + 0.0001;
  vec3 DiffuseColor = pow(texture(_MainTexture, Vertex.UV0).rgb, vec3(Gamma));
  vec3 SpecularColor = mix(vec3(1), _Material.Color, Metalness);
  
  vec3 F0 = vec3(SpecularColor * 0.04); 
  F0 = mix(F0, DiffuseColor, Metalness);
  vec3 Fresnel = FresnelSchlickRoughness(max(dot(Normal, EyeDirection), 0.0), F0, Roughness);
  vec2 EnviromentBRDF  = texture(_BRDFLUT, vec2(max(dot(Normal, EyeDirection), 0.0), Roughness)).rg;
  vec3 Irradiance = vec3(textureLod(_EnviromentMap, WorldReflection, _EnviromentMapLods - 2));
       Irradiance = dot(Irradiance, vec3(0.2126, 0.7152, 0.0722) / 2) * Irradiance;
  vec3 EnviromentLight = vec3(textureLod(_EnviromentMap, WorldReflection, Roughness * (_EnviromentMapLods - 4)));

  vec3 Specular = EnviromentLight * (Fresnel * EnviromentBRDF.x + EnviromentBRDF.y);

  // vec3 Color = (kD * DiffuseColor * EnviromentLight) + Specular;
  vec3 Color = (1 - (Fresnel * Metalness)) * DiffuseColor * Irradiance + Specular;
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
  float NDotV =   abs( dot( Normal, EyeDirection )) + 0.001; // Avoids artifact
  float HDotV = clamp( dot( HalfWayDirection, EyeDirection ), 0, 1 );

  float Roughness = _Material.Roughness * texture(_RoughnessTexture, Vertex.UV0).r + 0.0001;
  float Metalness = _Material.Metalness * texture(_MetallicTexture, Vertex.UV0).r;
  float Attenuation = SmoothDistanceAttenuation(UnormalizedLightDirection, _Lights[LightIndex].Intencity);

  vec3 SpecularColor = mix(vec3(1), _Material.Color, Metalness);
  vec3 LightColor = pow(_Lights[LightIndex].Color, vec3(Gamma));
  vec3 DiffuseColor = pow(texture(_MainTexture, Vertex.UV0).rgb, vec3(Gamma));
  vec3 F0 = vec3(SpecularColor * 0.04);
  F0 = mix(F0, DiffuseColor, Metalness);
  
  float NormalDistribution = TrowbridgeReitzNormalDistribution(NDotH, Roughness);
  float GeometricShadow = GeometrySmithSchlickGGX(NDotL, NDotV, Roughness);
  vec3  Fresnel = clamp(FresnelSchlick(HDotV, F0), 0, 1);

  vec3 Specular = (NormalDistribution * Fresnel * GeometricShadow) / (4 * (NDotL * NDotV) + 0.001);
  vec3 SurfaceColor = ((1 - (Fresnel * Metalness)) * DiffuseColor / PI + Specular) * Attenuation * LightColor * NDotL * NDotL;

  return SurfaceColor;
}

void main() {  
  vec3 Sum = vec3(0);

  for( int i = 0; i < 2; i++ ) {
    Sum += MicrofacetModel(i, Vertex.Position.xyz, Vertex.NormalDirection);
  }
  Sum += MicrofacetModelEnviroment(Vertex.Position.xyz, Vertex.NormalDirection);

  Sum = Sum / (Sum + vec3(1.0));
  Sum = pow(Sum, vec3(1.0/Gamma));

  FragColor = vec4(Sum, 1);

  if (FragColor.x < 0) {
    FragColor *= Matrix.Model * vec4(0);
  }
}
