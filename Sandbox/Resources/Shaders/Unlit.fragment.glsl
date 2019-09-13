#version 410

const float PI = 3.1415926535;
const float Gamma = 2.2;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform vec3 _ViewPosition;

uniform sampler2D _MainTexture;

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
  vec4 Diffuse = texture(_MainTexture, Vertex.UV0);
  vec3 DiffuseColor = pow(Diffuse.rgb, vec3(Gamma)) * _Material.Color.xyz;
  vec3 Color = _Material.Color.xyz * DiffuseColor;

  Color = Color / (Color + vec3(1.0));
  Color = pow(Color, vec3(1.0/Gamma));
  
  vec3 Intensity = vec3(dot(Color, vec3(0.2125, 0.7154, 0.0721)));
  Color = mix(Intensity, Color, 1.45);

  FragColor.xyz = Color;
  FragColor.a = _Material.Color.a * Vertex.Color.a * Diffuse.a;
  
  if (FragColor.x < 0) {
    FragColor *= Matrix.Model * Vertex.Color;
  }
}
