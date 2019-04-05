#version 410

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
  vec2 UV0;
  vec4 Color;
} Vertex;

void main() {
	Matrix.Model = _iModelMatrix;
	Matrix.WorldNormal = transpose(inverse(_iModelMatrix));

 	Vertex.Position = vec4(_iVertexPosition, 1.0);
 	Vertex.NormalDirection = normalize(Matrix.WorldNormal * vec4( _iVertexNormal, 1.0 )).xyz; 
	Vertex.UV0 = _iVertexUV0; 
	Vertex.Color = _iVertexColor;
  	
  	// Now set the position in model space
  	gl_Position = _ProjectionMatrix * _ViewMatrix * _iModelMatrix * Vertex.Position;
  	Vertex.Position = _iModelMatrix * Vertex.Position;
}
