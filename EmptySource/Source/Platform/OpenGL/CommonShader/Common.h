#include "Platform/OpenGL/CommonShader/VertexLayout.h"
#include "Platform/OpenGL/CommonShader/BasicStructs.h"
#include "Platform/OpenGL/CommonShader/BasicUniforms.h"
#include "Platform/OpenGL/CommonShader/VertexCommon.h"
#include "Platform/OpenGL/CommonShader/LightCommon.h"

namespace ESource {

	enum class EShaderToken {
		CommonVertex = 0,
		VertexLayout,
		VertexLayoutInstancing,
		Matrices,
		Vertex,
		Uniforms,
		Lights,
		UniformLights,
		Material,
		Max
	};

    struct GLSLShaderToken {
        const EShaderToken Token;
        const NChar * Name;
		const NString & VertexCode;
		const NString & PixelCode;
	};

    const GLSLShaderToken GLSLShaderTokens[(size_t)EShaderToken::Max] = {
		{ EShaderToken::CommonVertex,           "ESOURCE_COMMON_VERTEX",            CommonVertex,          ""                    },
		{ EShaderToken::VertexLayout,           "ESOURCE_VERTEX_LAYOUT",            VertexLayout,          ""                    },
		{ EShaderToken::VertexLayoutInstancing, "ESOURCE_VERTEX_LAYOUT_INSTANCING", VertexLayoutIntancing, ""                    },
		{ EShaderToken::Matrices,               "ESOURCE_MATRICES",                 CommonOutputMatrices,  CommonInputMatrices   },
		{ EShaderToken::Vertex,                 "ESOURCE_VERTEX",                   CommonOutputVertex,    CommonInputVertex     },
		{ EShaderToken::Uniforms,               "ESOURCE_UNIFORMS",                 CommonUniforms,        CommonUniforms        },
		{ EShaderToken::UniformLights,          "ESOURCE_UNIFORMLIGHTS",            CommonLight,           CommonLight           },
		{ EShaderToken::Lights,                 "ESOURCE_LIGHTS",                   CommonOutputLight,     CommonInputLight      },
		{ EShaderToken::Material,               "ESOURCE_MATERIAL",                 CommonUniformMaterial, CommonUniformMaterial },
    };

}