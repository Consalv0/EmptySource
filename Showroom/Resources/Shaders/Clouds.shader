Name: "CloudsShader"
Parameters:
  - Uniform: _EnviromentMap
    Type: Cubemap
    DefaultValue: {}
  - Uniform: _EnviromentMapLods
    Type: Float
    DefaultValue: 0.0
  - Uniform: _FlowMap
    Type: Texture2D
    DefaultValue: "NormalTexture"
  - Uniform: _FlowContribution
    Type: Float
    DefaultValue: 10.0
  - Uniform: _Scale
    Type: Float
    DefaultValue: 0.025
  - Uniform: _Speed
    Type: Float3D
    DefaultValue: [5.0, 0.0, 1.0]
    IsColor: false
  - Uniform: _BaseScale
    Type: Float
    DefaultValue: 0.01
  - Uniform: _BaseSpeed
    Type: Float3D
    DefaultValue: [1.6, 0.0, 0.5]
    IsColor: false
  - Uniform: _BaseStrength
    Type: Float
    DefaultValue: 0.888
  - Uniform: _OffsetScale
    Type: Float
    DefaultValue: -27.5
  - Uniform: _ColorValley
    Type: Float4D
    DefaultValue: [0.5, 0.7, 0.95, 0.0]
    IsColor: true
  - Uniform: _ColorPeak
    Type: Float4D
    DefaultValue: [0.99, 1.0, 1.0, 1.0]
    IsColor: true
  - Uniform: _NoiseRemaping
    Type: Float4D
    DefaultValue: [0.98, -0.24, 3.0, -0.28]
    IsColor: false
  - Uniform: _NoiseEdge
    Type: Float2D
    DefaultValue: [-0.65, 1.475]
    IsColor: true
  - Uniform: _NoisePower
    Type: Float
    DefaultValue: 1.31516
GLSL:
  Stages:
    - StageType: Vertex
      Code: |
        const float PI = 3.1415926535;
        
        ESOURCE_VERTEX_LAYOUT

        uniform mat4 _ModelMatrix;
        uniform mat4 _ProjectionMatrix;
        uniform mat4 _ViewMatrix;

        uniform sampler2D _FlowMap;
        uniform float _FlowContribution;
        uniform float _GlobalTime;
        uniform float _Scale;
        uniform vec3 _Speed;
        uniform float _BaseScale;
        uniform float _BaseStrength;
        uniform vec3 _BaseSpeed;
        uniform float _OffsetScale;
        uniform vec4 _ColorValley;
        uniform vec4 _ColorPeak;
        uniform vec4 _NoiseRemaping;
        uniform vec2 _NoiseEdge;
        uniform float _NoisePower;

        out struct Matrices {
          mat4 Model;
          mat4 WorldNormal;
        } Matrix;

        out struct VertexData {
          vec4 Position;
          vec3 NormalDirection;
          vec3 TangentDirection;
          vec3 BitangentDirection;
          vec2 UV0;
          vec4 Color;
        } Vertex;

        //	Classic Perlin3D Noise 
        //	by Stefan Gustavson
        vec4 Premute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
        vec4 TaylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
        vec3 Fade(vec3 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

        float Perlin3D(vec3 P) {
          vec3 Pi0 = floor(P); // Integer part for indexing
          vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
          Pi0 = mod(Pi0, 289.0);
          Pi1 = mod(Pi1, 289.0);
          vec3 Pf0 = fract(P); // Fractional part for interpolation
          vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
          vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
          vec4 iy = vec4(Pi0.yy, Pi1.yy);
          vec4 iz0 = Pi0.zzzz;
          vec4 iz1 = Pi1.zzzz;

          vec4 ixy = Premute(Premute(ix) + iy);
          vec4 ixy0 = Premute(ixy + iz0);
          vec4 ixy1 = Premute(ixy + iz1);

          vec4 gx0 = ixy0 / 7.0;
          vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
          gx0 = fract(gx0);
          vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
          vec4 sz0 = step(gz0, vec4(0.0));
          gx0 -= sz0 * (step(0.0, gx0) - 0.5);
          gy0 -= sz0 * (step(0.0, gy0) - 0.5);

          vec4 gx1 = ixy1 / 7.0;
          vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
          gx1 = fract(gx1);
          vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
          vec4 sz1 = step(gz1, vec4(0.0));
          gx1 -= sz1 * (step(0.0, gx1) - 0.5);
          gy1 -= sz1 * (step(0.0, gy1) - 0.5);

          vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
          vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
          vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
          vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
          vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
          vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
          vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
          vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

          vec4 norm0 = TaylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
          g000 *= norm0.x;
          g010 *= norm0.y;
          g100 *= norm0.z;
          g110 *= norm0.w;
          vec4 norm1 = TaylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
          g001 *= norm1.x;
          g011 *= norm1.y;
          g101 *= norm1.z;
          g111 *= norm1.w;

          float n000 = dot(g000, Pf0);
          float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
          float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
          float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
          float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
          float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
          float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
          float n111 = dot(g111, Pf1);

          vec3 fade_xyz = Fade(Pf0);
          vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
          vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
          float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
          return 2.2 * n_xyz;
        }

        float remap(float value, float min1, float max1, float min2, float max2) {
          return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
        }

        vec4 DeformPosition(vec4 Coord) {
          vec4 FlowVal = (texture(_FlowMap, Vertex.UV0) * 2.0 - 1.0) * vec4(_Speed, 0.0);
          float FlowDiff = sin(_GlobalTime * 0.25);
          vec3 DistortedPosition = (Matrix.Model * Coord).xyz - vec3(FlowVal.xy * FlowDiff, 0.0);
        	vec3 CoordV = DistortedPosition - (_Speed * _GlobalTime);
        	float V = (Perlin3D(CoordV * _Scale) + Perlin3D(DistortedPosition * _Scale)) * 0.5;
        	vec3 CoordW = (DistortedPosition + vec3(PI, PI, PI)) - (_BaseSpeed * _GlobalTime);
        	float W = Perlin3D(CoordW * _BaseScale);
         	W = W * _BaseStrength;
        	V = pow(clamp(abs(V), 0.0, 1.0), _NoisePower);
        	V = remap(V, _NoiseRemaping.x, _NoiseRemaping.y, _NoiseRemaping.z, _NoiseRemaping.w);
         	V = smoothstep(_NoiseEdge.r, _NoiseEdge.g, V);
            float CloudNoise = (W + V) / (1.0 + _BaseStrength);
          	vec4 NoiseOffset = vec4(_iVertexNormal * CloudNoise * _OffsetScale, 0.0);
        	return Coord + NoiseOffset;
        }

        void main() {
        	Matrix.Model = _ModelMatrix;
        	Matrix.WorldNormal = transpose(inverse(Matrix.Model));

         	Vertex.Position = vec4(_iVertexPosition, 1.0);
         	Vertex.NormalDirection = normalize(Matrix.WorldNormal * vec4( _iVertexNormal, 1.0 )).xyz; 
         	Vertex.TangentDirection = normalize(Matrix.WorldNormal * vec4( _iVertexTangent, 1.0 )).xyz;
        	Vertex.TangentDirection = normalize(Vertex.TangentDirection - dot(Vertex.TangentDirection, Vertex.NormalDirection) * Vertex.NormalDirection);
        	Vertex.BitangentDirection = cross(Vertex.NormalDirection, Vertex.TangentDirection);
        	Vertex.UV0 = _iVertexUV0; 
        	Vertex.Color = _iVertexColor;

        	Vertex.Position = DeformPosition(Vertex.Position);
          	Vertex.Position = Matrix.Model * Vertex.Position;

          	// Now set the position in model space
          	gl_Position = _ProjectionMatrix * _ViewMatrix * Matrix.Model * Vertex.Position;
        }
    - StageType: Pixel
      Code: |
        const float PI = 3.1415926535;
        const float Gamma = 2.2;
              
        uniform mat4 _ProjectionMatrix;
        uniform mat4 _ViewMatrix;
        uniform vec3 _ViewPosition;
              
        uniform samplerCube _EnviromentMap;
        uniform float _EnviromentMapLods;
              
        uniform sampler2D _FlowMap;
        uniform float _FlowContribution;
        uniform float _GlobalTime;
        uniform float _Scale;
        uniform vec3 _Speed;
        uniform float _BaseScale;
        uniform vec3 _BaseSpeed;
        uniform float _BaseStrength;
        uniform vec4 _ColorValley;
        uniform vec4 _ColorPeak;
        uniform vec4 _NoiseRemaping;
        uniform vec2 _NoiseEdge;
        uniform float _NoisePower;
              
        in struct Matrices {
          mat4 Model;
          mat4 WorldNormal;
        } Matrix;
              
        in struct VertexData {
          vec4 Position;
          vec3 NormalDirection;
          vec3 TangentDirection;
          vec3 BitangentDirection;
          vec2 UV0;
          vec4 Color;
        } Vertex;
              
        //	Classic Perlin3D Noise 
        //	by Stefan Gustavson
        vec4 Premute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
        vec4 TaylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
        vec3 Fade(vec3 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}
              
        float Perlin3D(vec3 P){
          vec3 Pi0 = floor(P); // Integer part for indexing
          vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
          Pi0 = mod(Pi0, 289.0);
          Pi1 = mod(Pi1, 289.0);
          vec3 Pf0 = fract(P); // Fractional part for interpolation
          vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
          vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
          vec4 iy = vec4(Pi0.yy, Pi1.yy);
          vec4 iz0 = Pi0.zzzz;
          vec4 iz1 = Pi1.zzzz;
              
          vec4 ixy = Premute(Premute(ix) + iy);
          vec4 ixy0 = Premute(ixy + iz0);
          vec4 ixy1 = Premute(ixy + iz1);
              
          vec4 gx0 = ixy0 / 7.0;
          vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
          gx0 = fract(gx0);
          vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
          vec4 sz0 = step(gz0, vec4(0.0));
          gx0 -= sz0 * (step(0.0, gx0) - 0.5);
          gy0 -= sz0 * (step(0.0, gy0) - 0.5);
              
          vec4 gx1 = ixy1 / 7.0;
          vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
          gx1 = fract(gx1);
          vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
          vec4 sz1 = step(gz1, vec4(0.0));
          gx1 -= sz1 * (step(0.0, gx1) - 0.5);
          gy1 -= sz1 * (step(0.0, gy1) - 0.5);
              
          vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
          vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
          vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
          vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
          vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
          vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
          vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
          vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);
              
          vec4 norm0 = TaylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
          g000 *= norm0.x;
          g010 *= norm0.y;
          g100 *= norm0.z;
          g110 *= norm0.w;
          vec4 norm1 = TaylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
          g001 *= norm1.x;
          g011 *= norm1.y;
          g101 *= norm1.z;
          g111 *= norm1.w;
              
          float n000 = dot(g000, Pf0);
          float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
          float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
          float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
          float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
          float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
          float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
          float n111 = dot(g111, Pf1);
              
          vec3 fade_xyz = Fade(Pf0);
          vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
          vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
          float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
          return 2.2 * n_xyz;
        }
              
        float remap(float value, float min1, float max1, float min2, float max2) {
          return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
        }
              
        vec3 FresnelSchlick(float CosTheta, vec3 F0) {
          return F0 + (1.0 - F0) * pow(1.0 - CosTheta, 5.0);
        }
              
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
          float Distribution = NdotH*NdotH * (RoughnessSqr - 1.0) + 1.0;
          return RoughnessSqr / (PI * Distribution*Distribution);
        }
              
        float SmoothDistanceAttenuation (vec3 UnormalizedLightVector, float AttenuationRadius) {
          float SquaredDistance = dot(UnormalizedLightVector, UnormalizedLightVector * 0.25) * (1.0 / AttenuationRadius);
              
          float Attenuation = 1.0 / ( max(SquaredDistance, 0.001) );
          return Attenuation;
        }
              
        out vec4 FragColor;
              
        void main() {
          vec3 Sum = vec3(0);
              
          Sum = Sum / (Sum + vec3(1.0));
          Sum = pow(Sum, vec3(1.0/Gamma));
              
          vec3 Intensity = vec3(dot(Sum, vec3(0.2125, 0.7154, 0.0721)));
          Sum = mix(Intensity, Sum, 1.45);
              
          vec4 FlowVal = (texture(_FlowMap, Vertex.UV0) * 2.0 - 1.0) * vec4(_Speed, 0.0);
          float FlowDiff0 = fract(_GlobalTime * 0.25);
          float FlowDiff1 = fract(_GlobalTime * 0.25 + 0.5);
          float LerpVal = abs((0.5 - FlowDiff0) / 0.5);
        	vec3 CoordW = (Vertex.Position.xyz + vec3(PI, PI, PI)) - (_BaseSpeed * _GlobalTime);
        	float W = Perlin3D(CoordW * _BaseScale);
          vec3 DistortedPosition = vec3(mix(FlowVal.xy * FlowDiff0, FlowVal.xy * FlowDiff1, LerpVal) * _FlowContribution, 0.0) * (W + 1.0) * 0.5 * _BaseStrength;
        	W = Perlin3D((CoordW + DistortedPosition) * _BaseScale);
        	vec3 CoordV = Vertex.Position.xyz - (_Speed * _GlobalTime);
        	float V = (Perlin3D(CoordV * _Scale) + Perlin3D(DistortedPosition * _Scale)) * 0.5;
          W = W * _BaseStrength;
        	V = pow(clamp(abs(V), 0.0, 1.0), _NoisePower);
        	V = remap(V, _NoiseRemaping.x, _NoiseRemaping.y, _NoiseRemaping.z, _NoiseRemaping.w);
         	V = smoothstep(_NoiseEdge.r, _NoiseEdge.g, V);
          float CloudNoise = (W + V) / (1.0 + _BaseStrength);
              
          FragColor = mix(_ColorValley, _ColorPeak, CloudNoise);
              
          if (FragColor.x == 0.001) {
            FragColor *= Matrix.Model * vec4(0);
          }
        }
