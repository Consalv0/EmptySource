#version 410

uniform sampler2D _MainTexture;
uniform vec2 _MainTextureSize;
uniform float _Time;
uniform float _Lod;

in struct VertexData {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec3 BitangentDirection;
  vec2 UV0;
  vec4 Color;
} vVertex;

out vec4 FragColor;

void main(){
    vec4 Color = textureLod(_MainTexture, vVertex.UV0, _Lod);
    FragColor = Color;
} 
