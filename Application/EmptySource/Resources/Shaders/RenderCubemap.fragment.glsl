#version 410

uniform samplerCube _Skybox;
uniform float _Time;

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
    vec4 Color = texture(_Skybox, normalize((Vertex.Position).xyz));
    FragColor = Color;
} 
