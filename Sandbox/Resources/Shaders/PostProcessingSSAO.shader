Name: "PostProcessingSSAOShader"
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
        
        uniform sampler2D _GPosition;
        uniform sampler2D _GNormal;
        uniform sampler2D _NoiseTexture;
        uniform vec2 _NoiseScale;
        uniform vec3 _Samples[64];
        
        const int KernelSize = 64;
        float Radius = 1.5;
        float Bias = 0.025;
        
        in vec2 UV0Coords;

        out vec4 FragColor;
        
        void main() {
          // get input for SSAO algorithm
          vec3 fragPos = texture(_GPosition, UV0Coords).xyz;
          vec3 normal = normalize(texture(_GNormal, UV0Coords).rgb);
          vec3 randomVec = normalize(texture(_NoiseTexture, UV0Coords * _NoiseScale).xyz);
          // create TBN change-of-basis matrix: from tangent-space to view-space
          vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
          vec3 bitangent = cross(normal, tangent);
          mat3 TBN = mat3(tangent, bitangent, normal);
          // iterate over the sample kernel and calculate occlusion factor
          float occlusion = 0.0;
          for (int i = 0; i < KernelSize; ++i) {
              // get sample position
              vec3 osample = TBN * _Samples[i]; // from tangent to view-space
              osample = fragPos + osample * Radius; 

              // project sample position (to sample texture) (to get position on screen/texture)
              vec4 offset = vec4(osample, 1.0);
              offset = _ProjectionMatrix * offset; // from view to clip-space
              offset.xyz /= offset.w; // perspective divide
              offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0

              // get sample depth
              float sampleDepth = texture(_GPosition, offset.xy).z; // get depth value of kernel sample

              // range check & accumulate
              float rangeCheck = smoothstep(0.0, 1.0, Radius / abs(fragPos.z - sampleDepth));
              occlusion += (sampleDepth >= osample.z + Bias ? 1.0 : 0.0) * rangeCheck;           
          }

          occlusion = 1.0 - (occlusion / KernelSize);

          FragColor = vec4(occlusion, occlusion, occlusion, 1.0);
        }
      