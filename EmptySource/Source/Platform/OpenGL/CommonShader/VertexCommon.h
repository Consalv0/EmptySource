const ESource::NString CommonVertex = R"(
ESOURCE_VERTEX_LAYOUT
ESOURCE_UNIFORMS
ESOURCE_MATRICES
ESOURCE_VERTEX

void ESource_VertexCompute() {
    Matrices.Model = _ModelMatrix;
    Matrices.WorldNormal = transpose(inverse(Matrices.Model));
    
    Vertex.Position = vec4(_iVertexPosition, 1.0);
    Vertex.NormalDirection = normalize(mat3(Matrices.WorldNormal) * _iVertexNormal); 
    Vertex.TangentDirection = normalize(mat3(Matrices.WorldNormal) * _iVertexTangent);
    Vertex.TangentDirection = normalize(Vertex.TangentDirection - dot(Vertex.TangentDirection, Vertex.NormalDirection) * Vertex.NormalDirection);
    Vertex.BitangentDirection = cross(Vertex.NormalDirection, Vertex.TangentDirection);
    Vertex.UV0 = _iVertexUV0; 
    Vertex.Color = _iVertexColor;

    Vertex.Position = Matrices.Model * Vertex.Position;
    Vertex.ScreenPosition = _ProjectionMatrix * _ViewMatrix * Vertex.Position;
}
)";
