#version 410

uniform samplerCube _Skybox;
uniform float _Lod;
uniform float _Gamma;

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
  vec3 BitangentDirection;
  vec2 UV0;
  vec4 Color;
} Vertex;

out vec4 FragColor;

void main(){
  vec3 Color = textureLod(_Skybox, normalize((Vertex.Position).xyz), _Lod).xyz;
  Color = Color / (Color + vec3(1.0));
  Color = pow(Color, vec3(1.0/_Gamma));
  vec3 Intensity = vec3(dot(Color, vec3(0.2125, 0.7154, 0.0721)));
  Color = mix(Intensity, Color, 1.45);
  FragColor = vec4(Color, 1.0);
  
  if (FragColor.x < 0) {
    FragColor *= Matrix.Model * vec4(0);
  }
} 
