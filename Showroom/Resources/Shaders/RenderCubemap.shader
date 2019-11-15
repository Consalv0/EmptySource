Name: "RenderCubemapShader"
Parameters:
  - Uniform: _Gamma
    Type: Float
    DefaultValue: 2.2
GLSL:
  Stages:
    - StageType: Vertex
      Code: |
        ESOURCE_VERTEX_LAYOUT

        uniform mat4 _ModelMatrix;
        uniform mat4 _ProjectionMatrix;
        uniform mat4 _ViewMatrix;
    
        out struct Matrices {
          mat4 Model;
          mat4 ModelWithoutPosition;
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
          Matrix.ModelWithoutPosition = _ModelMatrix;
          Matrix.ModelWithoutPosition[3][0] = 0.0;
          Matrix.ModelWithoutPosition[3][1] = 0.0;
          Matrix.ModelWithoutPosition[3][2] = 0.0;
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
          Vertex.Position = Matrix.ModelWithoutPosition * Vertex.Position;
        }
    - StageType: Pixel
      Code: |
        uniform samplerCube _Skybox;
        uniform float _Lod;
        uniform float _Gamma;

        uniform mat4 _ProjectionMatrix;
        uniform mat4 _ViewMatrix;

        in struct Matrices {
          mat4 Model;
          mat4 ModelWithoutPosition;
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

        void main(){
          vec3 Color = textureLod(_Skybox, normalize((Vertex.Position).xyz), _Lod).xyz;
          
          FragColor = vec4(Color, 1.0);

          if (FragColor.x < 0) {
            FragColor *= Matrix.Model * vec4(0);
          }
        } 
