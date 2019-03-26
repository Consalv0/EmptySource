#version 410

layout(location = 0) in vec3 _iVertexPosition;
layout(location = 6) in mat4 _iModelMatrix;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform vec3 _ViewPosition;
uniform sampler3D _Texture3D;

out struct Matrices {
  mat4 Model;
  mat4 WorldNormal;
} vMatrix;

out struct VertexSpace {
  vec4 Position;
} vVertex;

void main() {
	vMatrix.Model = _iModelMatrix;
	vMatrix.WorldNormal = transpose(inverse(_iModelMatrix));

 	vVertex.Position = vec4(_iVertexPosition, 1.0);
  	
  // Now set the position in model space
  vVertex.Position = _iModelMatrix * vVertex.Position;
  gl_Position = _ProjectionMatrix * _ViewMatrix * vVertex.Position;
}
