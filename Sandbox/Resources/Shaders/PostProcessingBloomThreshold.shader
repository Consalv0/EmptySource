Name: "PostProcessingBloomShader"
Parameters:
  - Uniform: _MainTexture
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