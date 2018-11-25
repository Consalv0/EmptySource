#version 460

layout(location = 0) in vec3 _iVertexPosition;
layout(location = 1) in vec3 _iVertexColor;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform mat4 _ModelMatrix;

out vec4 VertexPosition;
out vec3 Color;

void main() {
 	VertexPosition = vec4(_iVertexPosition, 1.0);
	Color = _iVertexColor;    

  	// Now set the position in model space
  	VertexPosition = _ModelMatrix * VertexPosition;
  	gl_Position = _ProjectionMatrix * _ViewMatrix * VertexPosition;
}