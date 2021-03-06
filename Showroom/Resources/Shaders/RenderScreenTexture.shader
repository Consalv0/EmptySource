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
        ESOURCE_VERTEX_LAYOUT

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

        vec3 GammaCorrection(vec3 Color) {
        	return pow(Color, vec3(1.0/_Gamma));
        }
        
        vec3 ReinhardTonemap(vec3 Color) {
          return Color / (Color + vec3(1.0));
        }

        vec3 Uncharted2Tonemap(vec3 Color) {
          float A = 0.15;
        	float B = 0.50;
        	float C = 0.10;
        	float D = 0.20;
        	float E = 0.02;
        	float F = 0.30;

          return ((Color*(A*Color+C*B)+D*E)/(Color*(A*Color+B)+D*F))-E/F;
        }
        
        void main() {
          vec4 Sample = texture(_MainTexture, UV0Coords, 0);
          vec4 SampleBloom = texture(_BloomTexture, UV0Coords, 0);
          float AmbientOcclusion = texture(_AOTexture, UV0Coords, 0).r;
          float Fog = texture(_DepthTexture, UV0Coords, 0).r;

          vec3 Color = Sample.rgb;
          Color *= pow(AmbientOcclusion, 7.0);

          vec3 BloomColor = SampleBloom.rgb;
          BloomColor = vec3(1.0) - exp(-BloomColor * _Exposure);
          BloomColor = clamp(BloomColor, vec3(0.0), vec3(1.0));

          vec3 BloomIntensity = vec3(dot(BloomColor, vec3(0.2125, 0.7154, 0.0721)));
          BloomColor = mix(BloomIntensity, BloomColor, 1 / 1.25);

          Color = Color + BloomColor * 0.5;
          Color = mix(Color, vec3(0.92, 0.76, 0.35) * 2, clamp(pow(Fog, 1.0 / 0.004) * 0.9, 0.0, 1.0));

          Color = Uncharted2Tonemap(Color);

          Color = vec3(1.0) - exp(-Color * _Exposure);
          Color = GammaCorrection(Color);

          // Color = clamp(Color, vec3(0.0), vec3(1.0));

          // vec3 ColorIntensity = vec3(dot(Color, vec3(0.2125, 0.7154, 0.0721)));
          // Color = mix(ColorIntensity, Color, 1.25);

          FragColor = vec4(Color, Sample.a * SampleBloom.a);
        }