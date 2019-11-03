Name: "GBufferPassShader"
Parameters:
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
  - Uniform: _EmissionTexture
    Type: Texture2D
    DefaultValue: "BlackTexture"
  - Uniform: _EmissionColor
    Type: Float3D
    DefaultValue: [ 0, 0, 0 ]
    IsColor: true
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

        uniform mat4 _ModelMatrix;
        uniform mat4 _ProjectionMatrix;
        uniform mat4 _ViewMatrix;

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

          Vertex.Position = _ViewMatrix * Matrix.Model * Vertex.Position;
          gl_Position = _ProjectionMatrix * Vertex.Position;

        }

    - StageType: Pixel
      Code: |
        const float PI = 3.1415926535;
        const float Gamma = 2.2;
        
        layout (location = 0) out vec3 GNormal;
        layout (location = 1) out vec4 GSpecular;
        
        uniform mat4 _ProjectionMatrix;
        uniform mat4 _ViewMatrix;
        uniform vec3 _ViewPosition;
        
        uniform sampler2D _MainTexture;
        uniform sampler2D _NormalTexture;
        uniform sampler2D _RoughnessTexture;
        uniform sampler2D _MetallicTexture;
        
        uniform struct MaterialInfo {
          vec4 Color;
        } _Material;

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

        // https://aras-p.info/texts/CompactNormalStorage.html
        vec2 EncodeStereographicProjection (vec3 Normal) {
          float Scale = 1.7777;
          vec2 Encode = Normal.xy / (Normal.z+1);
          Encode /= Scale;
          Encode = Encode * 0.5 + 0.5;
          return Encode;
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
          vec3 TangentNormal = texture(_NormalTexture, Vertex.UV0).rgb * 2.0 - 1.0;
          mat3 TBN = mat3(Vertex.TangentDirection, Vertex.BitangentDirection, Vertex.NormalDirection);
          vec3 VertNormal = normalize(TBN * TangentNormal);
          vec4 Diffuse = texture(_MainTexture, Vertex.UV0);
          float Metallic = texture(_MetallicTexture, Vertex.UV0).r;
          float Roughness = texture(_RoughnessTexture, Vertex.UV0).r;
          vec3 DiffuseColor = pow(Diffuse.rgb, vec3(Gamma)) * _Material.Color.xyz;

          GNormal.rg = EncodeStereographicProjection(VertNormal);
          // GAlbedo.rgb = DiffuseColor;
          // GAlbedo.a = Diffuse.a;
          GSpecular.r = Metallic;
          GSpecular.g = Roughness;
          GSpecular.zw = vec2(0.0, 1.0);
          
          if (Diffuse.a > 0.5)
            gl_FragDepth = gl_FragCoord.z;
          else
            gl_FragDepth = 1.0;
        }
      