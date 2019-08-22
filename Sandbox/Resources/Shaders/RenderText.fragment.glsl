#version 410

uniform sampler2D _MainTexture;
uniform vec2 _MainTextureSize;

uniform float _TextSize;
uniform float _TextBold;

in struct VertexData {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec3 BitangentDirection;
  vec2 UV0;
  vec4 Color;
} vVertex;

out vec4 FragColor;

float Contour(in float Distance, in float Width) {
    return smoothstep(_TextBold - Width, _TextBold + Width, Distance);
}

float Sample(in vec2 UV, float Width) {
    return Contour(texture(_MainTexture, UV).r, Width);
}

void main(){
  float Smoothing = 1 / _TextSize;
  float Distance = texture(_MainTexture, vVertex.UV0).r;
  float Width = fwidth(Distance) * 0.93 + Smoothing * 0.07;

  float Alpha = Contour( Distance, Width );
  
  // Supersample
  float DistScale = 0.354; // half of 1/sqrt2; you can play with this
  vec2 DistUV = DistScale * (dFdx(vVertex.UV0) + dFdy(vVertex.UV0));
  vec4 Box = vec4(vVertex.UV0 - DistUV, vVertex.UV0 + DistUV);
  float Sum = Sample( Box.xy, Width )
            + Sample( Box.zw, Width )
            + Sample( Box.xw, Width )
            + Sample( Box.zy, Width );
            
  Alpha = (Alpha + 0.5 * Sum) / 1.8;
  // Alpha = smoothstep(_TextBold - Smoothing, _TextBold + Smoothing, Distance);
  FragColor = vec4(1, 1, 1, Alpha);
} 
