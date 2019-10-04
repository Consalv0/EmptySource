Name: "PostProcessingBloomShader"
Parameters:
  - Uniform: _MainTextureThreshold
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
        const float BlurSize = 1.0/300.0;

        in vec2 UV0Coords;

        uniform float     _Radius;
        uniform vec2      _Direction;
        uniform vec2      _Resolution;
        uniform sampler2D _MainTexture;

        out vec4 FragColor;

        vec3 GaussianBlur13(sampler2D Texture, vec2 UV, vec2 Resolution, vec2 Direction, float Lod) {
          vec3 Color = vec3(0.0);
          vec2 off1 = vec2(1.4117647058823530) * Direction;
          vec2 off2 = vec2(3.2941176470588234) * Direction;
          vec2 off3 = vec2(5.1764705882352940) * Direction;
          Color += texture2D(Texture, UV, Lod).rgb * 0.1964825501511404;
          Color += texture2D(Texture, UV + (off1 / Resolution), Lod).rgb * 0.2969069646728344;
          Color += texture2D(Texture, UV - (off1 / Resolution), Lod).rgb * 0.2969069646728344;
          Color += texture2D(Texture, UV + (off2 / Resolution), Lod).rgb * 0.09447039785044732;
          Color += texture2D(Texture, UV - (off2 / Resolution), Lod).rgb * 0.09447039785044732;
          Color += texture2D(Texture, UV + (off3 / Resolution), Lod).rgb * 0.010381362401148057;
          Color += texture2D(Texture, UV - (off3 / Resolution), Lod).rgb * 0.010381362401148057;
          return Color;
        }

        void main() {
          vec3 Color = vec3(0.0);
          Color += GaussianBlur13(_MainTexture, UV0Coords, _Resolution, normalize(_Direction) * _Radius * 2.5, 5) * 0.1;
          Color += GaussianBlur13(_MainTexture, UV0Coords, _Resolution, normalize(_Direction) * _Radius * 1.5, 4) * 0.15;
          Color += GaussianBlur13(_MainTexture, UV0Coords, _Resolution, normalize(_Direction) * _Radius * 0.8, 3) * 0.33;
          Color += GaussianBlur13(_MainTexture, UV0Coords, _Resolution, normalize(_Direction) * _Radius * 0.2, 2) * 0.45;

          FragColor = vec4(Color, 1.0);
        }