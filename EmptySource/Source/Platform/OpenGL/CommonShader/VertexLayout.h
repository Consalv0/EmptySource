const ESource::NString VertexLayout = R"(
layout(location = 0) in vec3 _iVertexPosition;
layout(location = 1) in vec3 _iVertexNormal;
layout(location = 2) in vec3 _iVertexTangent;
layout(location = 3) in vec2 _iVertexUV0;
layout(location = 4) in vec2 _iVertexUV1;
layout(location = 5) in vec4 _iVertexColor;
)";

const ESource::NString VertexLayoutIntancing = R"(
layout(location = {}) in {} {};
)";