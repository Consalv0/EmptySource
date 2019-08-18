#version 410

uniform sampler2D _MainTexture;
uniform vec2 _MainTextureSize;
uniform int _Monochrome;
uniform vec4 _ColorFilter;
uniform float _Time;
uniform float _Lod;
uniform float _Gamma;

in struct VertexData {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec3 BitangentDirection;
  vec2 UV0;
  vec4 Color;
} vVertex;

out vec4 FragColor;

void main() {
    vec4 Color = textureLod(_MainTexture, vVertex.UV0, _Lod);
    Color.rgb = pow(Color.rgb, vec3(1.0/_Gamma));
  
    vec3 Intensity = vec3(dot(Color.rgb, vec3(0.2125, 0.7154, 0.0721)));
    Color.rgb = mix(Intensity, Color.rgb, 1.45);

    Color *= _ColorFilter;
    if (_Monochrome > 0) {
      float ColorLength = length(Color.rgb);
      if (_ColorFilter.a > 0.0) {
        ColorLength = length(Color);
      }
      Color = vec4(ColorLength, ColorLength, ColorLength, 1.0);
    }
    FragColor = Color;
} 
