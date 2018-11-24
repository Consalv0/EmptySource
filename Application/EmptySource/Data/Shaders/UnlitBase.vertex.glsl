#version 460

layout(location = 0) in vec3 _VertexPosition;
layout(location = 1) in vec3 _VertexColor;

out vec3 Color;

void main() {
	Color = _VertexColor;    
	gl_Position = vec4( _VertexPosition, 1.0 ); 
}