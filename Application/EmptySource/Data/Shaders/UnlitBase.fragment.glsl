#version 460 

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform mat4 _ModelMatrix;
uniform mat4 _WorldNormalMatrix;

in vec4 VertexPosition;
in vec3 NormalDirection;
in vec3 Color;
in vec2 TextureCoords;

out vec4 FragColor; 
 
void main() {
  vec3 NormalColor = (_ViewMatrix * vec4(NormalDirection, 1)).rgb;
  NormalColor = (1 - NormalColor) * 0.5;
  FragColor = vec4(NormalColor, 1.0);
} 