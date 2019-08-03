#version 410 core

layout(location = 0) in vec3 _iVertexPosition;
layout(location = 1) in vec3 _iVertexNormal;
layout(location = 2) in vec3 _iVertexTangent;
layout(location = 3) in vec2 _iVertexUV0;
layout(location = 4) in vec2 _iVertexUV1;
layout(location = 5) in vec4 _iVertexColor;
layout(location = 6) in mat4 _iModelMatrix;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;

out vec3 LocalPosition;

void main() {
    LocalPosition = _iVertexPosition;
    gl_Position = _ProjectionMatrix * _ViewMatrix * _iModelMatrix * vec4(LocalPosition, 1.0);
}