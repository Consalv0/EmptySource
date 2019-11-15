const ESource::NString CommonOutputMatrices = R"(
out struct ESourceMatrices {
  mat4 Model;
  mat4 WorldNormal;
} Matrices;
)";

const ESource::NString CommonInputMatrices = R"(
in struct ESourceMatrices {
  mat4 Model;
  mat4 WorldNormal;
} Matrices;
)";

const ESource::NString CommonOutputVertex = R"(
out struct ESourceVertex {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec3 BitangentDirection;
  vec2 UV0;
  vec4 Color;
  vec4 ScreenPosition;
} Vertex;
)";

const ESource::NString CommonInputVertex = R"(
in struct ESourceVertex {
  vec4 Position;
  vec3 NormalDirection;
  vec3 TangentDirection;
  vec3 BitangentDirection;
  vec2 UV0;
  vec4 Color;
  vec4 ScreenPosition;
} Vertex;
)";