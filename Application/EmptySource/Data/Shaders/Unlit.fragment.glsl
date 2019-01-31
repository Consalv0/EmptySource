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

void main() {
    FragColor = vec4(_Material.Color, 1);
}