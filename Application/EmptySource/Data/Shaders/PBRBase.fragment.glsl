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

// Fresnel term using the Schlick approximation
float SchlickFresnel(float i) {
    float x = clamp(1.0 - i, 0.0, 1.0);
    float x2 = x * x;
    return x2 * x2 * x;
}

float SchlickFresnelFuntion (float roughness, float NdotL, float NdotV, float LdotH) {
    float FresnelLight = SchlickFresnel(NdotL); 
    float FresnelView = SchlickFresnel(NdotV);
    float FresnelDiffuse90 = 0.5 + 2.0 * LdotH * LdotH * roughness;
    return mix(1, FresnelDiffuse90, FresnelLight) * mix(1, FresnelDiffuse90, FresnelView);
}

// Geometry function (G)
float ModifiedKelemenGeometricShadowingFunction (float roughness, float NdotV, float NdotL) {
	float c = 0.797884560802865; // sqrt(2 / Pi)
	float k = roughness * roughness * c;
	float gH = NdotV  * k + (1 - k);
	return (gH * gH * NdotL);
}

// The normal distribution function (D)
float GGXTrowbridgeReitzDistribution(float roughness, float NdotH) {
  float roughnessSqr = roughness * roughness;
  float distribution = NdotH * NdotH * (roughnessSqr - 1.0) + 1.0;
  return roughnessSqr / (PI * distribution * distribution);
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
  vec3 NormalColor = (_ViewMatrix * vec4(Normal, 1)).rgb;
       NormalColor = (1 - NormalColor) * 0.5;

  float LinearRoughness = _Material.Roughness;
  float Roughness = LinearRoughness * LinearRoughness;
  float Metalness = _Material.Metalness;

  vec3 DiffuseColor = NormalColor * (1 - Metalness);
       // DiffuseColor *= texture(_texture, uv).rgb;

  vec3 SpecularColor = mix(vec3(1), _Material.Color, Metalness);

  vec3 UnormalizedLightDirection = (_Lights[LightIndex].Position - VertPosition).xyz;
  float LightDistance = length(UnormalizedLightDirection);

  vec3 LightDirection = normalize(UnormalizedLightDirection);
  vec3 EyeDirection = normalize(_ViewPosition.xyz - VertPosition.xyz);
  vec3 HalfWayDirection = normalize(EyeDirection + LightDirection);
  
  float LDotH = clamp( dot( LightDirection, HalfWayDirection ), 0, 1 );
  float NDotH = clamp( dot( Normal, HalfWayDirection ), 0, 1 );
  float NDotL = clamp( dot( Normal, LightDirection ), 0, 1 );
  float NDotV = abs( dot (Normal , EyeDirection )) + 1e-5f; // avoid artifact

  float InvSqrLightRadius = 1 / ( (_Lights[LightIndex].Intencity * _Lights[LightIndex].Intencity) + 1e-5f) ;
  float Attenuation = 1;
        // Attenuation *= ShadowCalculation(positionLightSpace, 0.0001);
        // Attenuation *= (-dot(LightDirection * 2, LightDirection) - 1.4) * 6;
        Attenuation *= GetLightAttenuation(UnormalizedLightDirection, InvSqrLightRadius);

  vec3 Diffuse = NDotL * _Lights[LightIndex].Color * Attenuation;

  vec3 SpecularDistribution = _Lights[LightIndex].Color * SpecularColor * Attenuation;
       SpecularDistribution *= max( GGXTrowbridgeReitzDistribution(Roughness, NDotH), 0.0 );
  
  float GeometricShadow = max( ModifiedKelemenGeometricShadowingFunction(Roughness, NDotV, NDotL), 0.0 );
  
  float Fresnel = max( SchlickFresnelFuntion(Roughness, NDotL, NDotV, LDotH), 0.0 );

  vec3 Specularity = (SpecularDistribution * GeometricShadow * Fresnel) / (4 * ( NDotL * NDotV ));
  
  vec3 SurfaceColor = Diffuse * DiffuseColor;
       SurfaceColor += max( Specularity * NDotV, 0 );

  return SurfaceColor;
}

void main() {  
  vec3 Sum = vec3(0);

  for( int i = 0; i < 2; i++ ) {
    Sum += MicrofacetModel(i, Vertex.Position.xyz, Vertex.NormalDirection);
  }
 
  Sum = pow( Sum, vec3(1.0/Gamma) );

  FragColor = vec4(Sum, 1);
} 
