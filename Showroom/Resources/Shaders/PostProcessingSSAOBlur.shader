Name: "PostProcessingSSAOBlurShader"
Parameters:
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
        uniform mat4 _ProjectionMatrix;
        
        uniform sampler2D _SSAO;
        
        in vec2 UV0Coords;

        out vec4 FragColor;
        
        void main() {
          vec2 texelSize = 1.0 / vec2(textureSize(_SSAO, 0));
          float result = 0.0;
          for (int x = -2; x < 2; ++x) {
              for (int y = -2; y < 2; ++y) {
                vec2 offset = vec2(float(x), float(y)) * texelSize;
                result += texture(_SSAO, UV0Coords + offset).r;
              }
          }
          FragColor = vec4(result / (4.0 * 4.0));
          FragColor.a = 1.0;
        }
      