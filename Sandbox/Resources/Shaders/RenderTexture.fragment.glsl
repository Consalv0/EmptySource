#version 410

const vec2 InvAtan = vec2(0.1591, 0.3183);

uniform sampler2D _MainTexture;
uniform samplerCube _MainTextureCube;
uniform int _IsCubemap;
uniform int _Monochrome;
uniform vec4 _ColorFilter;
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

vec3 SampleEquirectangular(vec2 UV) {
    vec2 UVInv = (UV + vec2(0.5, 0) - 0.5) / InvAtan;
    float CosPhi = cos(UVInv.y);
    float SinPhi = sin(UVInv.y);
    float CosTheta = cos(UVInv.x);
    float SinTheta = sin(UVInv.x);
    vec3 Vector = vec3(CosTheta * CosPhi, SinPhi, SinTheta * CosPhi);
    return Vector;
}

void main() {
  vec4 Color;
  if (_IsCubemap > 0) {
    Color = textureLod(_MainTextureCube, SampleEquirectangular(vVertex.UV0), _Lod);
  } else {
    Color = textureLod(_MainTexture, vVertex.UV0, _Lod);
  }

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
