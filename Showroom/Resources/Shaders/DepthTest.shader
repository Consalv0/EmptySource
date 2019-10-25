Name: "DepthTestShader"
Parameters:
  - Uniform: _ViewMatrix
    Type: Matrix4x4
    DefaultValue: {}
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

          gl_Position = _ProjectionMatrix * _ViewMatrix * Matrix.Model * Vertex.Position;
          Vertex.Position = Matrix.Model * Vertex.Position;
        }
    - StageType: Pixel
      Code: |
        uniform sampler2D _MainTexture;

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

        void main() {
          vec4 Diffuse = texture(_MainTexture, Vertex.UV0);
          
          if (Diffuse.a > 0.5)
            gl_FragDepth = gl_FragCoord.z;
          else
            gl_FragDepth = 1.0;
        }
      