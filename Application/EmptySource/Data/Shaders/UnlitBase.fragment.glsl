#version 460

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;

in mat4 ModelMatrix;
in mat4 WorldNormalMatrix;

in vec4 VertexPosition;
in vec3 NormalDirection;
in vec2 UV0;
in vec4 Color;

out vec4 FragColor; 
 
void main() {
  vec3 NormalColor = (_ViewMatrix * vec4(NormalDirection, 1)).rgb;
  NormalColor = (1 - NormalColor) * 0.5;
  FragColor = vec4(NormalColor, 1.0);
} 