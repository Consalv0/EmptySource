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

/*
 * This shader is based in the spesification of https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec.md.html
*/

vec3 BSDFDiffuse() {

}

// The model is a sum of a diffuse, a reflective and a transmissive component:
vec4 Model( int LightIndex, vec3 VertPosition, vec3 VertNormal ) {
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
}

void main() {  
  vec3 Sum = vec3(0);

  for( int i = 0; i < 2; i++ ) {
    Sum += MicrofacetModel(i, Vertex.Position.xyz, Vertex.NormalDirection);
  }
 
  Sum = pow( Sum, vec3(1.0/Gamma) );

  FragColor = vec4(Sum, 1);
}
