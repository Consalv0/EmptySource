#version 410

const float PI = 3.1415926535;
const float Gamma = 2.2;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform vec3 _ViewPosition;

uniform sampler2D _MainTexture;
uniform sampler2D _NormalTexture;
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
  vec4 Color;          // Diffuse color for dielectrics, f0 for metallic
} _Material;

in struct Matrices {
  mat4 Model;
  mat4 WorldNormal;
} Matrix;

in struct VertexData {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec3 BitangentDirection;
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
  float SquaredDistance = dot(UnormalizedLightVector, UnormalizedLightVector * 0.25) * (1.0 / AttenuationRadius);

  float Attenuation = 1.0 / ( max(SquaredDistance, 0.001) );
  return Attenuation;
}

// Enviroment Light
vec3 MicrofacetModelEnviroment( vec3 VertPosition, vec3 Normal, float Roughness, float Metalness, vec3 DiffuseColor ) {
  vec3 EyeDirection = normalize(_ViewPosition.xyz - VertPosition.xyz);
  vec3 WorldReflection = reflect(-EyeDirection, normalize(Normal));
  
  float NDotV = clamp(abs( dot( Normal, EyeDirection ) ) + 0.01, 0.0, 1.0);

  vec3 F0 = vec3(0.2); 
  F0 = mix(F0, DiffuseColor, Metalness);
  vec3 Fresnel = FresnelSchlickRoughness(NDotV, F0, Roughness);
  vec2 EnviromentBRDF  = texture(_BRDFLUT, vec2(NDotV, Roughness)).rg;
  vec3 Irradiance = vec3(textureLod(_EnviromentMap, Normal, _EnviromentMapLods - 2.0));
       // Irradiance = sqrt(pow(Irradiance, vec3(1.0/2.2)));
  vec3 EnviromentLight = vec3(textureLod(_EnviromentMap, WorldReflection, Roughness * (_EnviromentMapLods - 3.0)));

  vec3 Specular = EnviromentLight * (Fresnel * EnviromentBRDF.x + EnviromentBRDF.y);
  vec3 Color = ((vec3(1.0) - Fresnel) * (1.0 - Metalness) * DiffuseColor / PI * Irradiance + Specular);

  return Color;
}

// Point Light Calculation
vec3 MicrofacetModel( int LightIndex, vec3 VertPosition, vec3 Normal, float Roughness, float Metalness, vec3 DiffuseColor ) {
  vec3 UnormalizedLightDirection = (_Lights[LightIndex].Position - VertPosition).xyz;
  float LightDistance = length(UnormalizedLightDirection);

  vec3 LightDirection = normalize(UnormalizedLightDirection);
  vec3 EyeDirection = normalize(_ViewPosition.xyz - VertPosition.xyz);
  vec3 HalfWayDirection = normalize(EyeDirection + LightDirection);
  
  float LDotH = clamp( dot( LightDirection, HalfWayDirection ), 0.0, 1.0 );
  float NDotH = clamp( dot( Normal, HalfWayDirection ), 0.0, 1.0 );
  float NDotL = clamp( dot( Normal, LightDirection ), 0.0, 1.0 );
  float NDotV = clamp( abs( dot( Normal, EyeDirection ) ) + 0.01, 0.0, 1.0);
  float HDotV = clamp( dot( HalfWayDirection, EyeDirection ), 0.0, 1.0 );

  float Attenuation = SmoothDistanceAttenuation(UnormalizedLightDirection, _Lights[LightIndex].Intencity);
  vec2 EnviromentBRDF  = texture(_BRDFLUT, vec2(NDotV, Roughness)).rg;
  vec3 LightColor = pow(_Lights[LightIndex].Color * (_Lights[LightIndex].Intencity / vec3(8125, 10154, 7721) * SmoothDistanceAttenuation(UnormalizedLightDirection, _Lights[LightIndex].Intencity * 0.001) + 1.0), vec3(Gamma));
       LightColor = max(LightColor, vec3(0, 0, 0));
  vec3 F0 = vec3(0.2);
  F0 = mix(F0, DiffuseColor, Metalness);
  
  float NormalDistribution = TrowbridgeReitzNormalDistribution(NDotH, Roughness);
  float GeometricShadow = GeometrySmithSchlickGGX(NDotL, NDotV, Roughness);
  vec3 Fresnel = FresnelSchlickRoughness(NDotV, F0, Roughness);

  vec3 Specular = NormalDistribution * GeometricShadow * LightColor * (Fresnel * EnviromentBRDF.x + EnviromentBRDF.y);
  vec3 SurfaceColor = ((vec3(1.0) - Fresnel) * (1.0 - Metalness) * DiffuseColor / PI + Specular) * Attenuation * LightColor * NDotL;

  return SurfaceColor;
}

void main() {  
  vec3 Sum = vec3(0);

  vec3 TangentNormal = texture(_NormalTexture, Vertex.UV0).rgb * 2.0 - 1.0;
  mat3 TBN = mat3(Vertex.TangentDirection, Vertex.BitangentDirection, Vertex.NormalDirection);
  vec3 VertNormal = normalize(TBN * TangentNormal);
  float Roughness = clamp(_Material.Roughness * texture(_RoughnessTexture, Vertex.UV0).r, 0.0001, 1.0);
  float Metalness = clamp(_Material.Metalness * texture(_MetallicTexture, Vertex.UV0).r, 0.0001, 1.0);
  vec4 Diffuse = texture(_MainTexture, Vertex.UV0);
  vec3 DiffuseColor = pow(Diffuse.rgb, vec3(Gamma)) * _Material.Color.xyz;

  for( int i = 0; i < 2; i++ ) {
    Sum += MicrofacetModel(i, Vertex.Position.xyz, VertNormal, Roughness, Metalness, DiffuseColor);
  }
  Sum += MicrofacetModelEnviroment(Vertex.Position.xyz, VertNormal, Roughness, Metalness, DiffuseColor);

  Sum = Sum / (Sum + vec3(1.0));
  Sum = pow(Sum, vec3(1.0/Gamma));
  
  vec3 Intensity = vec3(dot(Sum, vec3(0.2125, 0.7154, 0.0721)));
  Sum = mix(Intensity, Sum, 1.45);

  FragColor = vec4(Sum, Vertex.Color.a * Diffuse.a * _Material.Color.a);

  if (FragColor.x == 0.001) {
    FragColor *= Matrix.Model * vec4(0);
  }
}
