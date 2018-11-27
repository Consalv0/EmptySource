#version 460

layout(location = 0) in vec3 _iVertexPosition;
layout(location = 1) in vec3 _iVertexNormal;
layout(location = 2) in vec2 _iVertexTextureCoords;
layout(location = 3) in vec4 _iVertexColor;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform mat4 _ModelMatrix;
uniform mat4 _WorldNormalMatrix;

out vec4 VertexPosition;
out vec3 NormalDirection;
out vec2 TextureCoords;
out vec4 Color;

void main() {
 	VertexPosition = vec4(_iVertexPosition, 1.0);
 	NormalDirection = normalize(_WorldNormalMatrix * vec4( _iVertexNormal, 1.0 )).xyz; 
	TextureCoords = _iVertexTextureCoords; 
	Color = _iVertexColor;

  	// Now set the position in model space
  	VertexPosition = _ModelMatrix * VertexPosition;
  	gl_Position = _ProjectionMatrix * _ViewMatrix * VertexPosition;
}