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
    // FragColor = texture( 
    //     _MainTexture, vVertex.UV0 + 0.005 * vec2( sin(_Time + _MainTextureSize.x * vVertex.UV0.x), cos(_Time + _MainTextureSize.y * vVertex.UV0.y) )
    // );
    // FragColor = texture(_MainTexture, vVertex.UV0 + 0.05 * vec2(_Time, _Time * 0.5));
    FragColor = texture(_MainTexture, vVertex.UV0);
} 