#version 460

uniform mat4 _ProjectionMatrix;
uniform mat4 _ViewMatrix;
uniform vec3 _ViewPosition;

uniform struct LightInfo {
	vec3 Position;        // Light Position in camera coords.
	vec3 ColorIntencity;  // Light Color and Intensity
} _Lights[2];

uniform struct MaterialInfo {
  float Roughness;     // Roughness
  float Metalness;     // Metallic (0) or dielectric (1)
  vec3 SpecularColor;  // Diffuse color for dielectrics, f0 for metallic
} _Material;

in mat4 ModelMatrix;
in mat4 WorldNormalMatrix;

in vec4 VertexPosition;
in vec3 NormalDirection;
in vec3 TangentDirection;
in vec2 UV0;
in vec4 Color;

out vec4 FragColor; 


// Fresnel term using the Schlick approximation
float SchlickFresnel(float i){
    float x = clamp(1.0 - i, 0.0, 1.0);
    float x2 = x * x;
    return x2 * x2 * x;
}

float SchlickFresnelFuntion (float roughness, float NdotL, float NdotV, float LdotH) {
    float FresnelLight = SchlickFresnel(NdotL); 
    float FresnelView = SchlickFresnel(NdotV);
    float FresnelDiffuse90 = 0.5 + 2.0 * LdotH * LdotH * roughness;
    return mix(1, FresnelDiffuse90, FresnelLight) * mix(1, FresnelDiffuse90, FresnelView);
}

// Geometry function (G)
float ModifiedKelemenGeometricShadowingFunction (float roughness, float NdotV, float NdotL) {
	float c = 0.797884560802865; // sqrt(2 / Pi)
	float k = roughness * roughness * c;
	float gH = NdotV  * k + (1 - k);
	return (gH * gH * NdotL);
}

// The normal distribution function (D)
float GGXTrowbridgeReitzDistribution(float roughness, float NdotH) {
  float roughnessSqr = roughness * roughness;
  float distribution = NdotH * NdotH * (roughnessSqr - 1.0) + 1.0;
  return roughnessSqr / (3.1415926535 * distribution * distribution);
}

// Light Calculation
vec3 MicrofacetModel( int LightIndex, vec3 VertPosition, vec3 VertNormal ) {

  // vec3 DiffuseColor = mix(_Material.Color, vec3(0.0), _Material.Metalness); // Metallic Color
  vec3 DiffuseColor = _Material.SpecularColor.rgb * (1 - _Material.Metalness);
       // DiffuseColor *= texture(_texture, uv).rgb;

  vec3 SpecularColor = mix(vec3(1), _Material.SpecularColor, _Material.Metalness);

  vec3 LightDirection = (_Lights[LightIndex].Position - VertPosition).xyz;
  float LightDistance = length(LightDirection);

  LightDirection = normalize(LightDirection);
  vec3 HalfWayDirection = normalize( VertPosition + LightDirection );
  vec3 EyeDirection = normalize(_ViewPosition.xyz - VertPosition.xyz);
  
  float NDotH = max( 0.0, dot( VertNormal, HalfWayDirection ) );
  float LDotH = max( 0.0, dot( LightDirection, HalfWayDirection ) );
  float NDotL = max( 0.0, dot( VertNormal, LightDirection ) );
  float NDotV = max( 0.0, dot( VertNormal, EyeDirection ) );

  float LightIntencity = length(_Lights[LightIndex].ColorIntencity) / (LightDistance * LightDistance);
  float Attenuation = LightIntencity;
        // Attenuation *= ShadowCalculation(positionLightSpace, 0.0001);
        // Attenuation *= (-dot(LightDirection * 2, LightDirection) - 1.4) * 6;
        Attenuation = clamp(Attenuation, 0, 1);

  vec3 Diffuse = NDotL * _Lights[LightIndex].ColorIntencity * Attenuation;

  float Roughness = _Material.Roughness;

  vec3 SpecularDistribution = _Lights[LightIndex].ColorIntencity * _Material.SpecularColor * Attenuation;
       SpecularDistribution *= max( GGXTrowbridgeReitzDistribution(Roughness, NDotH), 0.0 );
  float GeometricShadow = max( ModifiedKelemenGeometricShadowingFunction(Roughness, NDotV, NDotL), 0.0 );
  float Fresnel = max( SchlickFresnelFuntion(Roughness, NDotL, NDotV, LDotH), 0.0 );

  vec3 Specularity = (SpecularDistribution * GeometricShadow * Fresnel) / (4 * ( NDotL * NDotV ));

  vec3 NormalColor = (_ViewMatrix * vec4(NormalDirection, 1)).rgb;
       NormalColor = (1 - NormalColor) * 0.5;
  
  vec3 SurfaceColor = Diffuse * NormalColor;
       SurfaceColor += max( Specularity * NDotV, 0 );

  return SurfaceColor;
}

void main() {  
  vec3 Sum = vec3(0);

  for( int i = 0; i < 2; i++ ) {
    Sum += MicrofacetModel(i, VertexPosition.xyz, NormalDirection);
  }

  // Gamma 
  Sum = pow( Sum, vec3(1.0/2.2) );

  FragColor = vec4(Sum, 1);
} 