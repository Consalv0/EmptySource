Name: "UnLitShader"
Parameters:
  - Uniform: _Material.Color
    Type: Float4D
    DefaultValue: [ 1, 1, 1, 1 ]
    IsColor: true
  - Uniform: _MainTexture
    Type: Texture2D
    DefaultValue: "WhiteTexture"
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
        	Matrix.Model = _iModelMatrix;
        	Matrix.WorldNormal = transpose(inverse(Matrix.Model));
    
         	Vertex.Position = vec4(_iVertexPosition, 1.0);
         	Vertex.NormalDirection = normalize(Matrix.WorldNormal * vec4( _iVertexNormal, 1.0 )).xyz; 
         	Vertex.TangentDirection = normalize(Matrix.WorldNormal * vec4( _iVertexTangent, 1.0 )).xyz;
        	Vertex.TangentDirection = normalize(Vertex.TangentDirection - dot(Vertex.TangentDirection, Vertex.NormalDirection) * Vertex.NormalDirection);
        	Vertex.BitangentDirection = cross(Vertex.NormalDirection, Vertex.TangentDirection);
        	Vertex.UV0 = _iVertexUV0; 
        	Vertex.Color = _iVertexColor;
    
          // Now set the position in model space
          gl_Position = _ProjectionMatrix * _ViewMatrix * Matrix.Model * Vertex.Position;
          Vertex.Position = Matrix.Model * Vertex.Position;
        }
    - StageType: Pixel
      Code: |
        const float PI = 3.1415926535;
        const float Gamma = 2.2;

        uniform mat4 _ProjectionMatrix;
        uniform mat4 _ViewMatrix;
        uniform vec3 _ViewPosition;

        uniform sampler2D _MainTexture;

        uniform struct LightInfo {
        	vec3 Position;        // Light Position in camera coords.
          vec3 Color;           // Light Color
        	float Intencity;      // Light Intensity
        } _Lights[2];

        uniform struct MaterialInfo {
          float Roughness;     // Roughness
          float Metalness;     // Metallic (0) or dielectric (1)
          vec4 Color;          // Diffuse color for dielectrics, f0 for metallic
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

        out vec4 FragColor;

        void main() {
          vec4 Diffuse = texture(_MainTexture, Vertex.UV0);
          vec3 DiffuseColor = pow(Diffuse.rgb, vec3(Gamma)) * _Material.Color.xyz;
          vec3 Color = _Material.Color.xyz * DiffuseColor;

          Color = Color / (Color + vec3(1.0));
          Color = pow(Color, vec3(1.0/Gamma));

          vec3 Intensity = vec3(dot(Color, vec3(0.2125, 0.7154, 0.0721)));
          Color = mix(Intensity, Color, 1.45);

          FragColor.xyz = Color;
          FragColor.a = _Material.Color.a * Vertex.Color.a * Diffuse.a;

          if (FragColor.x < 0) {
            FragColor *= Matrix.Model * Vertex.Color;
          }
        }
      