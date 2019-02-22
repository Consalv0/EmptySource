#version 460

uniform sampler2D _MainTexture;
uniform vec2 _MainTextureSize;
uniform float _Time;

in struct VertexData {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec2 UV0;
  vec4 Color;
} vVertex;

out vec4 FragColor;

void main(){
    FragColor = texture( 
        _MainTexture, vVertex.UV0
    );
} 