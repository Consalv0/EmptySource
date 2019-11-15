Name: "PostProcessingBloomShader"
Parameters:
  - Uniform: _MainTexture
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
        const float Gamma = 2.2;
        const float Exposure = 1.0;

        in vec2 UV0Coords;

        uniform float _Threshold;
        uniform sampler2D _MainTexture;
              
        out vec4 FragColor;
        
        void main() {
          vec4 Sample = texture(_MainTexture, UV0Coords, 0);

          vec3 Color = Sample.rgb;

          Color = mix(vec3(0.0), Color - vec3(0.5), min(dot(Color, vec3(0.2125, 0.7154, 0.0721)) * _Threshold, 3.0));

          FragColor = vec4(Color, 1.0);
        }