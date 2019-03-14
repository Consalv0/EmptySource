#version 460

uniform sampler2D _MainTexture;
uniform vec2 _MainTextureSize;
uniform float _Time;

uniform float _TextSize;
uniform float _TextBold;

in struct VertexData {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec2 UV0;
  vec4 Color;
} vVertex;

out vec4 FragColor;

void main(){
  float Smoothing = 1.0 / _TextSize;
  float Distance = texture(_MainTexture, vVertex.UV0).r;
  float Alpha = smoothstep(_TextBold - Smoothing, _TextBold + Smoothing, Distance);
  FragColor = vec4(Alpha, Alpha, Alpha, 0);
} 