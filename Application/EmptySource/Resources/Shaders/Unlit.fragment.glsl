#version 410

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

void main() {
  vec3 Color = normalize(_Material.Color.xyz);
  float Intensity = dot(_Material.Color.xyz, vec3(0.3, 0.7, 0.07));
  
  FragColor.xyz = pow(vec3(Intensity) * Color + vec3(Color), Intensity * vec3(0.3, 0.7, 0.07));
  FragColor.a = _Material.Color.a * Vertex.Color.a;
  
  if (FragColor.x < 0) {
    FragColor *= Matrix.Model * Vertex.Color;
  }
}
