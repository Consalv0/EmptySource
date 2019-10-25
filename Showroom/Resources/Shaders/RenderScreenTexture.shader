Name: "RenderScreenTextureShader"
Parameters:
  - Uniform: _MainTexture
    Type: Texture2D
    DefaultValue: "BlackTexture"
  - Uniform: _BloomTexture
    Type: Texture2D
    DefaultValue: "BlackTexture"
GLSL:
  Stages:
    - StageType: Vertex
      Code: |
        layout(location = 0) in vec3 _iVertexPosition;
        layout(location = 1) in vec3 _iVertexNormal;
        layout(location = 2) in vec3 _iVertexTangent;
        layout(location = 3) in vec2 _iVertexUV0;
        layout(location = 4) in vec2 _iVertexUV1;
        layout(location = 5) in vec4 _iVertexColor;

        out vec2 UV0Coords;

        void main() {
          UV0Coords = _iVertexUV0;
          gl_Position = vec4(_iVertexPosition, 1.0);
        }
    - StageType: Pixel
      Code: |
        in vec2 UV0Coords;

        uniform sampler2D _MainTexture;
        uniform sampler2D _BloomTexture;
        uniform sampler2D _AOTexture;
        uniform sampler2D _DepthTexture;
        uniform float _Gamma;
        uniform float _Exposure;
              
        out vec4 FragColor;
        
        void main() {
          vec4 Sample = texture(_MainTexture, UV0Coords, 0);
          vec4 SampleBloom = texture(_BloomTexture, UV0Coords, 0);
          float AmbientOcclusion = texture(_AOTexture, UV0Coords, 0).r;
          float Fog = texture(_DepthTexture, UV0Coords, 0).r;

          vec3 Color = Sample.rgb;
          Color *= pow(AmbientOcclusion, 5.0);

          Color = vec3(1.0) - exp(-Color * _Exposure);
          Color = pow(Color, vec3(1.0 / _Gamma));
          Color = clamp(Color, vec3(0.0), vec3(1.0));

          vec3 ColorIntensity = vec3(dot(Color, vec3(0.2125, 0.7154, 0.0721)));
          Color = mix(ColorIntensity, Color, 1.25);

          vec3 BloomColor = SampleBloom.rgb;
          BloomColor = vec3(1.0) - exp(-BloomColor * _Exposure);
          BloomColor = clamp(BloomColor, vec3(0.0), vec3(1.0));

          vec3 BloomIntensity = vec3(dot(BloomColor, vec3(0.2125, 0.7154, 0.0721)));
          BloomColor = mix(BloomIntensity, BloomColor, 1 / 1.25);

          Color = Color + BloomColor * 0.5;
          Color = mix(Color, vec3(0.95, 0.85, 0.8), clamp(pow(Fog, 1.0 / 0.006) * 1.15, 0.0, 1.0));

          FragColor = vec4(Color, Sample.a * SampleBloom.a);
        }