Name: "HDRClampingShader"
Parameters:
  - Uniform: _EquirectangularMap
    Type: Texture2D
    DefaultValue: "BlackTexture"
  - Uniform: _LodLevel
    Type: Float
    DefaultValue: 0.0
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
        layout(location = 6) in mat4 _iModelMatrix;

        uniform mat4 _ProjectionMatrix;

        out struct VertexData {
          vec4 Position;
          vec3 NormalDirection;
          vec3 TangentDirection;
          vec3 BitangentDirection;
          vec2 UV0;
          vec4 Color;
        } vVertex;

        void main() {
          vVertex.Position = vec4(_iVertexPosition, 1.0);

          vVertex.NormalDirection = normalize(vec4( _iVertexNormal, 1.0 )).xyz; 
          vVertex.TangentDirection = _iVertexTangent;
          vVertex.UV0 = _iVertexUV0;
          vVertex.Color = _iVertexColor;

          // Now set the position in model space
          gl_Position = _ProjectionMatrix * _iModelMatrix * vVertex.Position;
          vVertex.Position = _iModelMatrix * vVertex.Position;
        }
    - StageType: Pixel
      Code: |
        const float PI = 3.14159265359;
        const vec2 InvAtan = vec2(0.1591, 0.3183);

        uniform sampler2D _EquirectangularMap;
        uniform float _LodLevel;

        in struct VertexData {
          vec4 Position;
          vec3 NormalDirection;
          vec3 TangentDirection;
          vec3 BitangentDirection;
          vec2 UV0;
          vec4 Color;
        } vVertex;

        out vec4 FragColor;

        vec2 SampleSphericalMap(vec3 Vector) {
            vec2 UV = vec2(atan(Vector.z, Vector.x), asin(Vector.y));
            UV *= InvAtan;
            UV += 0.5;
            return UV;
        }

        vec3 SampleEquirectangular(vec2 UV) {
            vec2 UVInv = (UV - 0.5) / InvAtan;
            float CosPhi = cos(UVInv.y);
            float SinPhi = sin(UVInv.y);
            float CosTheta = cos(UVInv.x);
            float SinTheta = sin(UVInv.x);
            vec3 Vector = vec3(CosTheta * CosPhi, SinPhi, SinTheta * CosPhi);
            return Vector;
        }

        void main() {	
            vec3 N = normalize(SampleEquirectangular(vVertex.UV0));

            vec3 Irradiance = vec3(0.0);   

            // tangent space calculation from origin point
            vec3 UpDir    = vec3(0.0, 1.0, 0.0);
            vec3 RightDir = cross(UpDir, N);
                 UpDir    = cross(N, RightDir);

            float SampleDelta = 0.025;
            float SampleCount = 0.0;
            for(float Phi = 0.0; Phi < 2.0 * PI; Phi += SampleDelta) {
                for(float Theta = 0.0; Theta < 0.5 * PI; Theta += SampleDelta) {
                    float CosTheta = cos(Theta);
                    float SinTheta = sin(Theta);
                    // Spherical to cartesian (in tangent space)
                    vec3 TangentSample = vec3(SinTheta * cos(Phi),  SinTheta * sin(Phi), CosTheta);
                    // Tangent space to world
                    vec3 SampleVector = TangentSample.x * RightDir + TangentSample.y * UpDir + TangentSample.z * N; 

                    Irradiance += clamp(texture(_EquirectangularMap, SampleSphericalMap(SampleVector)).rgb, 0, 10000) * dot(N, SampleVector);
                    SampleCount++;
                }
            }
            Irradiance = PI * Irradiance / SampleCount;

            FragColor = vec4(clamp(texture(_EquirectangularMap, vVertex.UV0).rgb, 0, 10000), 1);
        }