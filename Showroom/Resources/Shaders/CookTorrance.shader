Name: "CookTorranceShader"
Parameters:
  - Uniform: _Material.Metalness
    Type: Float
    DefaultValue: 1.0
  - Uniform: _Material.Roughness
    Type: Float
    DefaultValue: 1.0
  - Uniform: _Material.Color
    Type: Float4D
    DefaultValue: [ 1, 1, 1, 1 ]
    IsColor: true
  - Uniform: _MainTexture
    Type: Texture2D
    DefaultValue: "WhiteTexture"
  - Uniform: _NormalTexture
    Type: Texture2D
    DefaultValue: "NormalTexture"
  - Uniform: _RoughnessTexture
    Type: Texture2D
    DefaultValue: "WhiteTexture"
  - Uniform: _MetallicTexture
    Type: Texture2D
    DefaultValue: "BlackTexture"
  - Uniform: _AOTexture
    Type: Texture2D
    DefaultValue: "WhiteTexture"
  - Uniform: _EmissionTexture
    Type: Texture2D
    DefaultValue: "BlackTexture"
  - Uniform: _EmissionColor
    Type: Float3D
    DefaultValue: [ 0, 0, 0 ]
    IsColor: true
  - Uniform: _BRDFLUT
    Type: Texture2D
    DefaultValue: "BlackTexture"
  - Uniform: _EnviromentMap
    Type: Cubemap
    DefaultValue: {}
  - Uniform: _EnviromentMapLods
    Type: Float
    DefaultValue: 0.0
GLSL:
  Stages:
    - StageType: Vertex
      Code: |
        ESOURCE_VERTEX_LAYOUT_INSTANCING(1,mat4,_ModelMatrix)
        ESOURCE_COMMON_VERTEX
        ESOURCE_LIGHTS
    
        void main() {
          ESource_VertexCompute();
          ESource_ComputeLights();

          gl_Position = Vertex.ScreenPosition;
        }

    - StageType: Pixel
      Code: |
        const float PI = 3.1415926535;
        const float Gamma = 2.2;
        
        ESOURCE_MATRICES
        ESOURCE_VERTEX
        ESOURCE_UNIFORMS
        ESOURCE_LIGHTS
        ESOURCE_MATERIAL

        uniform float       _EnviromentMapLods;
        uniform samplerCube _EnviromentMap;
        uniform sampler2D   _BRDFLUT;
        
        out vec4 FragColor;
      
        vec3 FresnelSchlickRoughness(float CosTheta, vec3 F0, float Roughness) {
          return F0 + (max(vec3(1.0 - Roughness), F0) - F0) * pow(1.0 - CosTheta, 5.0);
        }   
      
        float GeometrySmithSchlickGGX(float NDotL, float NDotV, float Roughness) {
          float RoughnessPlusOne = (Roughness);
          float k = (RoughnessPlusOne * RoughnessPlusOne) / 8.0;
      
          float GGXDotV = NDotV / ( NDotV * (1.0 - k) + k );
          float GGXDotL = NDotL / ( NDotL * (1.0 - k) + k );
      
          return GGXDotV * GGXDotL;
        }
      
        float TrowbridgeReitzNormalDistribution(float NdotH, float Roughness){
          float RoughnessSqr = Roughness * Roughness;
          float NdotHSqr = NdotH * NdotH;
          float Distribution = NdotHSqr * (RoughnessSqr - 1.0) + 1.0;
          return RoughnessSqr / (PI * Distribution * Distribution);
        }
      
        float SmoothDistanceAttenuation (vec3 UnormalizedLightVector, float AttenuationRadius) {
          float SquaredDistance = dot(UnormalizedLightVector, UnormalizedLightVector * 0.25) * (1.0 / AttenuationRadius);
      
          float Attenuation = 1.0 / ( max(SquaredDistance, 0.001) );
          return Attenuation;
        }
        
        float ShadowCalculation(int LightIndex, float NDotL) {
          vec3 ProjCoords = Lights.LightSpacePosition[LightIndex].xyz / Lights.LightSpacePosition[LightIndex].w;
          vec2 UVCoords;
          UVCoords.x = 0.5 * ProjCoords.x + 0.5;
          UVCoords.y = 0.5 * ProjCoords.y + 0.5;
          float z = 0.5 * ProjCoords.z + 0.5;

          float xOffset = 1.0/1024.0;
          float yOffset = 1.0/1024.0;
          float Step = 0.5;
          float Iterations = 0.0;

          float Factor = 0.0;

          for (float y = -1 ; y <= 1 ; y += Step) {
              for (float x = -1 ; x <= 1 ; x += Step) {
                  vec2 Offsets = vec2(x * xOffset, y * yOffset);
                  vec2 UVC = vec2(UVCoords + Offsets);
                  float ClosestDepth = texture(_Lights[LightIndex].ShadowMap, UVC).r;
                  float CurrentDepth = z - _Lights[LightIndex].ShadowBias;
                  float Shadow = step(CurrentDepth, ClosestDepth);
                  Shadow *= step(z, 1.0);
                  Factor += Shadow;
                  Iterations += 1.0;
              }
          }

          return Factor / Iterations / 2.0;
        }
      
        // Enviroment Light
        vec3 MicrofacetModelEnviroment( vec3 VertPosition, vec3 Normal, float Roughness, float Metalness, vec3 DiffuseColor ) {
          vec3 EyeDirection = normalize(_ViewPosition.xyz - VertPosition.xyz);
          vec3 WorldReflection = reflect(-EyeDirection, normalize(Normal));
      
          float NDotV = clamp(abs( dot( Normal, EyeDirection ) ) + 0.01, 0.0, 1.0);
      
          // F0 - base reflectivity of a surface, a.k.a. surface reflection at zero incidence
          // or how much the surface reflects if looking directly at it
          vec3 F0 = vec3(0.04); // 0.04 is a commonly used constant for dielectrics
          F0 = mix(F0, DiffuseColor, Metalness);

          vec3 Fresnel = FresnelSchlickRoughness(NDotV, F0, Roughness);

          vec2 EnviromentBRDF = texture(_BRDFLUT, vec2(NDotV, Roughness)).rg;
          vec3 Irradiance = vec3(textureLod(_EnviromentMap, Normal, _EnviromentMapLods - 2.0));
               //Irradiance = pow(Irradiance, vec3(1.0/dot(vec3(0.2125, 0.7154, 0.0721) * PI * 4.0, Irradiance)));
          vec3 EnviromentLight = vec3(textureLod(_EnviromentMap, WorldReflection, Roughness * (_EnviromentMapLods - 3.0)));
      
          vec3 Specular = EnviromentLight * (Fresnel * EnviromentBRDF.x + EnviromentBRDF.y);
          vec3 Color = ((1.0 - Metalness) * DiffuseColor * Irradiance + Specular);
      
          return Color;
        }
      
        // Spot Light Calculation
        vec3 MicrofacetModel( int LightIndex, vec3 VertPosition, vec3 Normal, float Roughness, float Metalness, vec3 DiffuseColor ) {
          vec3 UnormalizedLightDirection = (_Lights[LightIndex].Position - VertPosition).xyz;
          float LightDistance = length(UnormalizedLightDirection);
      
          vec3 LightDirection = normalize(UnormalizedLightDirection);
          vec3 EyeDirection = normalize(_ViewPosition.xyz - VertPosition.xyz);
          vec3 HalfWayDirection = normalize(EyeDirection + LightDirection);
      
          float LDotH = clamp( dot( LightDirection, HalfWayDirection ), 0.0, 1.0 );
          float NDotH = clamp( dot( Normal, HalfWayDirection ), 0.0, 1.0 );
          float NDotL = clamp( dot( Normal, LightDirection ), 0.0, 1.0 );
          float NDotV = clamp( abs( dot( Normal, EyeDirection ) ) + 0.01, 0.0, 1.0);
          float HDotV = clamp( dot( HalfWayDirection, EyeDirection ), 0.0, 1.0 );
      
          float SpotDirection = clamp((dot(-_Lights[LightIndex].Direction * 2.0, LightDirection) - 1.4) * 4, 0.0, 1.0);
                // SpotDirection = TrowbridgeReitzNormalDistribution(SpotDirection, 0.6);
          float ShadowMapTest = ShadowCalculation(LightIndex, NDotL);
          float Attenuation = SmoothDistanceAttenuation(UnormalizedLightDirection, _Lights[LightIndex].Intencity * SpotDirection);
                Attenuation *= ShadowMapTest;
          vec2 EnviromentBRDF  = texture(_BRDFLUT, vec2(NDotV, Roughness)).rg;
          vec3 LightColor = pow(_Lights[LightIndex].Color * (_Lights[LightIndex].Intencity / vec3(8125, 10154, 7721) * SmoothDistanceAttenuation(UnormalizedLightDirection, _Lights[LightIndex].Intencity * 0.001) + 1.0), vec3(Gamma));
               LightColor = max(LightColor, vec3(0, 0, 0));
          vec3 F0 = vec3(0.2);
          F0 = mix(F0, DiffuseColor, Metalness);
      
          float NormalDistribution = TrowbridgeReitzNormalDistribution(NDotH, Roughness);
          float GeometricShadow = GeometrySmithSchlickGGX(NDotL, NDotV, Roughness);
          vec3 Fresnel = FresnelSchlickRoughness(NDotV, F0, Roughness);
      
          vec3 Specular = NormalDistribution * GeometricShadow * LightColor * (Fresnel * EnviromentBRDF.x + EnviromentBRDF.y);
          vec3 SurfaceColor = ((vec3(1.0) - Fresnel) * (1.0 - Metalness) * DiffuseColor / PI + Specular) * Attenuation * LightColor * NDotL;

          return SurfaceColor;
        }
      
        void main() {  
          vec3 Sum = vec3(0);
      
          vec3 TangentNormal = texture(_NormalTexture, Vertex.UV0).rgb * 2.0 - 1.0;
          mat3 TBN = mat3(Vertex.TangentDirection, Vertex.BitangentDirection, Vertex.NormalDirection);
          vec3 VertNormal = normalize(TBN * TangentNormal);
          float Roughness = clamp(_Material.Roughness * texture(_RoughnessTexture, Vertex.UV0).r, 0.0001, 1.0);
          float Metalness = clamp(_Material.Metalness * texture(_MetallicTexture, Vertex.UV0).r, 0.0001, 1.0);
          vec4 Diffuse = texture(_MainTexture, Vertex.UV0);
          vec3 DiffuseColor = pow(Diffuse.rgb, vec3(Gamma)) * _Material.Color.xyz;
          vec3 Emission = texture(_EmissionTexture, Vertex.UV0).rgb * _EmissionColor;
          float AmbientOclussion = texture(_AOTexture, Vertex.UV0).r;
      
          for( int i = 0; i < 2; i++ ) {
            Sum += MicrofacetModel(i, Vertex.Position.xyz, VertNormal, Roughness, Metalness, DiffuseColor);
          }
          Sum += MicrofacetModelEnviroment(Vertex.Position.xyz, VertNormal, Roughness, Metalness, DiffuseColor) * AmbientOclussion;
          Sum += Emission;

          FragColor = vec4(Sum, Vertex.Color.a * Diffuse.a * _Material.Color.a);
          
          if (FragColor.a > 0.5)
            gl_FragDepth = gl_FragCoord.z;
          else
            gl_FragDepth = 1.0;
          
          if (FragColor.x == 0.001) {
            FragColor *= Matrices.Model * vec4(0);
          }
        }
      