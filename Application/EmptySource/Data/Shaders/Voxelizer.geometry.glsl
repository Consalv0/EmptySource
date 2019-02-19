#version 460

// Input is triangles, output is triangle strip. Because we're going
// to do a 1 in 1 out shader producing a single triangle output for
// each one input.
layout (points) in;
layout (triangle_strip, max_vertices=36) out;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform vec3 _ViewPosition;
uniform sampler3D _Texture3D;

in struct Matrices {
  mat4 Model;
  mat4 WorldNormal;
} vMatrix[];

in struct VertexSpace {
  vec4 Position;
} vVertex[];

out Matrices Matrix;

out struct VertexData {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec2 UV0;
  vec4 Color;
} Vertex;

const vec4 CubeVertices[8] = vec4[8] (
    vec4( 0.5, -0.5, -0.5, 0),
    vec4( 0.5, -0.5,  0.5, 0),
    vec4(-0.5, -0.5,  0.5, 0),
    vec4(-0.5, -0.5, -0.5, 0),
    vec4( 0.5,  0.5, -0.5, 0),
    vec4( 0.5,  0.5,  0.5, 0),
    vec4(-0.5,  0.5,  0.5, 0),
    vec4(-0.5,  0.5, -0.5, 0)
);

const int CubeIndices[36] = int[36] (
    2, 4, 1, 8, 6, 5,
    5, 2, 1, 6, 3, 2,
    3, 8, 4, 1, 8, 5,
    2, 3, 4, 8, 7, 6,
    5, 6, 2, 6, 7, 3,
    3, 7, 8, 1, 4, 8
);

void BuildCube() {
    for (int i_face = 0; i_face < 12; i_face++) {
        for (int i_vert = 0; i_vert < 3; i_vert++) {
  	        gl_Position = _ProjectionMatrix * _ViewMatrix * (round(vVertex[0].Position) + CubeVertices[CubeIndices[i_face * 3 + i_vert] - 1]);
            Matrix.Model       = vMatrix[0].Model;
            Matrix.WorldNormal = vMatrix[0].WorldNormal;
            Vertex.Position        = vVertex[0].Position + CubeVertices[CubeIndices[i_face * 3 + i_vert] - 1];
            Vertex.NormalDirection = texture(_Texture3D, vVertex[0].Position.xyz).xyz;// vVertex[1].NormalDirection;
            Vertex.UV0             = texture(_Texture3D, vVertex[0].Position.xyz).xy; // vVertex[1].UV0; 
            Vertex.Color           = texture(_Texture3D, vVertex[0].Position.xyz);    // vVertex[1].Color;
            EmitVertex();
        }

        EndPrimitive();
    }
}

void main() {
    BuildCube();
    EndPrimitive();
}