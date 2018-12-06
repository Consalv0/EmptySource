#version 460

layout(location = 0) in vec3 _iVertexPosition;
layout(location = 1) in vec3 _iVertexNormal;
layout(location = 2) in vec3 _iVertexTangent;
layout(location = 3) in vec2 _iVertexUV0;
layout(location = 4) in vec2 _iVertexUV1;
layout(location = 5) in vec4 _iVertexColor;
layout(location = 6) in mat4 _iModelMatrix;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;

out mat4 ModelMatrix;
out mat4 WorldNormalMatrix;

out vec4 VertexPosition; 
out vec3 NormalDirection;
out vec3 TangentDirection;
out vec2 UV0;
out vec4 Color;

void main() {
 	VertexPosition = vec4(_iVertexPosition, 1.0);
	ModelMatrix = _iModelMatrix;
	WorldNormalMatrix = transpose(inverse(_iModelMatrix));
 	NormalDirection = normalize(WorldNormalMatrix * vec4( _iVertexNormal, 1.0 )).xyz; 
	UV0 = _iVertexUV0; 
	Color = _iVertexColor;

  	// Now set the position in model space
  	VertexPosition = _iModelMatrix * VertexPosition;
  	gl_Position = _ProjectionMatrix * _ViewMatrix * VertexPosition;
}