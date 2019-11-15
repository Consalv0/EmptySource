Name: "IntegrateBRDFShader"
Parameters:
GLSL:
  Stages:
    - StageType: Vertex
      Code: |
        ESOURCE_VERTEX_LAYOUT

        uniform mat4 _ModelMatrix;
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
          gl_Position = _ProjectionMatrix * _ModelMatrix * vVertex.Position;
          vVertex.Position = _ModelMatrix * vVertex.Position;
        }
    - StageType: Pixel
      Code: |
        const float PI = 3.14159265359;
        const vec2 InvAtan = vec2(0.1591, 0.3183);

        in struct VertexData {
          vec4 Position;
          vec3 NormalDirection;
          vec3 TangentDirection;
          vec3 BitangentDirection;
          vec2 UV0;
          vec4 Color;
        } vVertex;

        out vec2 FragColor;

        float RadicalInverse_VdC(uint bits) {
            bits = (bits << 16u) | (bits >> 16u);
            bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
            bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
            bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
            bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
            return float(bits) * 2.3283064365386963e-10; // / 0x100000000
        }

        vec2 Hammersley(uint i, uint N) {
            return vec2(float(i)/float(N), RadicalInverse_VdC(i));
        }

        vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float Roughness) {
            float RoughnessSqr = Roughness * Roughness;

            float Phi = 2.0 * PI * Xi.x;
            float CosTheta = sqrt((1.0 - Xi.y) / (1.0 + (RoughnessSqr*RoughnessSqr - 1.0) * Xi.y));
            float SinTheta = sqrt(1.0 - CosTheta*CosTheta);

            // --- From spherical coordinates to cartesian coordinates
            vec3 H;
            H.x = SinTheta * cos(Phi);
            H.y = SinTheta * sin(Phi);
            H.z = CosTheta;

            // --- From tangent-space vector to world-space sample vector
            vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            vec3 TangentX = normalize( cross(UpVector, N) );
            vec3 TangentY = cross( N, TangentX );

            vec3 SampleVec = TangentX * H.x + TangentY * H.y + N * H.z;
            return normalize(SampleVec);
        } 

        float WalterEtAlGeometricShadowingFunction (float NDotL, float NDotV, float RoughnessSqr){
            float NDotLSqr = NDotL*NDotL;
            float NDotVSqr = NDotV*NDotV;

            float SmithLight = 2/(1 + sqrt(1 + RoughnessSqr * (1-NDotLSqr)/(NDotLSqr)));
            float SmithVisibility = 2/(1 + sqrt(1 + RoughnessSqr * (1-NDotVSqr)/(NDotVSqr)));

        	float GeometricShadow = (SmithLight * SmithVisibility);
        	return GeometricShadow;
        }

        vec2 IntegrateBRDF(float NDotV, float Roughness) {
            vec3 V;
            V.x = sqrt(1.0 - NDotV * NDotV);
            V.y = 0;
            V.z = NDotV;

            float A = 0;
            float B = 0;

            vec3 N = vec3(0.0, 0.0, 1.0);

            const uint NumSamples = 1024u;
            for(uint i = 0u; i < NumSamples; ++i)
            {
                vec2 Xi = Hammersley(i, NumSamples);
                vec3 H  = ImportanceSampleGGX(Xi, N, Roughness);
                vec3 L  = normalize(2.0 * dot(V, H) * H - V);

                float NDotL = max( L.z, 0 );
                float NDotH = max( H.z, 0 );
                float VDotH = max( dot( V, H ), 0 );

                if(NDotL > 0.0) {
                    float G = WalterEtAlGeometricShadowingFunction(NDotL, NDotV, Roughness * Roughness);
                    float G_Vis = G * VDotH / (NDotH * NDotV);
                    float Fc = pow(1 - VDotH, 5);

                    A += (1 - Fc) * G_Vis;
                    B += Fc * G_Vis;
                }
            }

            return vec2(A, B) / NumSamples;
        }

        void main() {
            vec2 IntegratedBRDF = IntegrateBRDF(vVertex.UV0.x, vVertex.UV0.y);
            FragColor = IntegratedBRDF;
        }
