#version 460

// Input is triangles, output is triangle strip. Because we're going
// to do a 1 in 1 out shader producing a single triangle output for
// each one input.
layout (triangles) in;
layout (triangle_strip, max_vertices=36) out;

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform vec3 _ViewPosition;

in struct Matrices {
  mat4 Model;
  mat4 WorldNormal;
} vMatrix[];

in struct VertexData {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec2 UV0;
  vec4 Color;
} vVertex[];

out Matrices Matrix;
out VertexData Vertex;

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

void BuildCube(int index) {
    for (int i_face = 0; i_face < 12; i_face++) {
        for (int i_vert = 0; i_vert < 3; i_vert++) {
  	        gl_Position = _ProjectionMatrix * _ViewMatrix * (round(vVertex[index].Position) + CubeVertices[CubeIndices[i_face * 3 + i_vert] - 1]);
            Matrix.Model       = vMatrix[index].Model;
            Matrix.WorldNormal = vMatrix[index].WorldNormal;
            Vertex.Position        = vVertex[index].Position + CubeVertices[CubeIndices[i_face * 3 + i_vert] - 1];
            Vertex.NormalDirection = vVertex[index].NormalDirection;
            Vertex.UV0             = vVertex[index].UV0; 
            Vertex.Color           = vVertex[index].Color;
            EmitVertex();
        }

        EndPrimitive();
    }
}

void main() {
    BuildCube(1);

    EndPrimitive();
}