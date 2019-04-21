#version 410

uniform samplerCube _Skybox;
uniform float _Roughness;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;

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

void main(){
    vec3 Color = textureLod(_Skybox, normalize((Vertex.Position).xyz), _Roughness).xyz;
    Color = Color / (Color + vec3(1.0));
    Color = pow(Color, vec3(1.0/1.8));
    FragColor = vec4(Color, 1.0);
} 
