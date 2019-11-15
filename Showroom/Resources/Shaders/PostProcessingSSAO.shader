Name: "PostProcessingSSAOShader"
Parameters:
GLSL:
  Stages:
    - StageType: Vertex
      Code: |
        ESOURCE_VERTEX_LAYOUT

        uniform float _AspectRatio;
        uniform float _TanHalfFOV;

        out vec2 UV0Coords;
        out vec2 ViewRay;

        void main() {
          UV0Coords = (_iVertexPosition.xy + vec2(1.0)) / 2.0;
          ViewRay.x = _iVertexPosition.x;
          ViewRay.y = _iVertexPosition.y;
          gl_Position = vec4(_iVertexPosition, 1.0);
        }
    - StageType: Pixel
      Code: |
        const int SAMPLE_COUNT = 16;		
        const int KERNEL_SIZE = 64;
        
        uniform mat4 _ProjectionMatrix;
        uniform mat4 _ViewMatrix;
        
        uniform sampler2D _GDepth;
        uniform sampler2D _GNormal;
        uniform sampler2D _NoiseTexture;
        uniform vec2 _NoiseScale;
        uniform vec3 _Kernel[KERNEL_SIZE];

        float Radius = 1.5;
        float Bias = 0.025;
        
        in vec2 UV0Coords;
        in vec2 ViewRay;

        out float FragColor;
        
        float CalcViewZ(vec2 Coords) {
            float Depth = texture(_GDepth, Coords).x;
            float A     = _ProjectionMatrix[2][2];
            float B     = _ProjectionMatrix[3][2];
            float NDC_z = 2.0 * Depth - 1.0;
            float ViewZ = B / (A + NDC_z);
            return ViewZ;
        }
        
        vec3 DecodeStereographicProjection (vec2 Encode) {
          float Scale = 1.7777;
          vec3 nn =
              vec3(Encode.xy, 0.0) * vec3(2.0*Scale, 2.0*Scale, 0.0) +
              vec3(-Scale, -Scale, 1.0);
          float g = 2.0 / dot(nn.xyz,nn.xyz);
          vec3 Normal;
          Normal.xy = g * nn.xy;
          Normal.z = g - 1.0;
          return Normal;
        }

        void main() {

          float ViewZ = CalcViewZ(UV0Coords);
          float ViewX = ViewRay.x * ViewZ / _ProjectionMatrix[0][0];
          float ViewY = ViewRay.y * ViewZ / _ProjectionMatrix[1][1];

          vec3 Position = vec3(ViewX, ViewY, -ViewZ);
          vec3 Normal = mat3(_ViewMatrix) * DecodeStereographicProjection(normalize(texture(_GNormal, UV0Coords).rg));
          vec3 RandomVec = normalize(texture(_NoiseTexture, UV0Coords * _NoiseScale).xyz);
          // create TBN change-of-basis matrix: from tangent-space to view-space
          vec3 Tangent = normalize(RandomVec - Normal * dot(RandomVec, Normal));
          vec3 BiTangent = cross(Normal, Tangent);
          mat3 TBN = mat3(Tangent, BiTangent, Normal);

          float Occlusion = 0.0;

          for (int i = 0 ; i < SAMPLE_COUNT ; i++) {
              vec3 SamplePos = TBN * _Kernel[i];
                   SamplePos = Position + SamplePos * Radius;
              vec4 Offset = vec4(SamplePos, 1.0);
              Offset = _ProjectionMatrix * Offset;
              Offset.xy /= Offset.w;
              Offset.xy = Offset.xy * 0.5 + vec2(0.5);

              float SampleDepth = -CalcViewZ(Offset.xy);
              
              // if (abs(Position.z - SampleDepth) < Radius) {
              //   Occlusion += step(SampleDepth, SamplePos.z);
              // }
              float RangeCheck = smoothstep(0.0, 1.0, Radius / abs(Position.z - SampleDepth));
              Occlusion += (SampleDepth >= SamplePos.z + Bias ? 1.0 : 0.0) * RangeCheck;   
          }

          Occlusion = 1.0 - (Occlusion / KERNEL_SIZE);
          Occlusion = pow(Occlusion, 1.0);
          FragColor = Occlusion;
        }
      