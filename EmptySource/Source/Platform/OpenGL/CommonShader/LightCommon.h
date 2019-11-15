const ESource::NString CommonLight = R"(
uniform struct ESourceLightInfo {
    vec3 Position;
    vec3 Color;
    float Intencity;
    vec3 Direction;
    mat4 ViewMatrix;
    mat4 ProjectionMatrix;
    sampler2D ShadowMap;
    float ShadowBias;
} _Lights[2];
)";

const ESource::NString CommonOutputLight = R"(
ESOURCE_UNIFORMLIGHTS
out struct ESourceLight {
  vec4 LightSpacePosition[2];
} Lights;

void ESource_ComputeLights() {
    for (int i = 0; i < 2; i++) {
        Lights.LightSpacePosition[i] = _Lights[i].ProjectionMatrix * _Lights[i].ViewMatrix * Vertex.Position;
    }
};
)";

const ESource::NString CommonInputLight = R"(
ESOURCE_UNIFORMLIGHTS
in struct ESourceLight {
  vec4 LightSpacePosition[2];
} Lights;
)";
