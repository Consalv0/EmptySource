
#include "CoreMinimal.h"
#include "Rendering/Rendering.h"
#include "Resources/ShaderResource.h"
#include "Rendering/Texture.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderingAPI.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"

#include "Rendering/MeshPrimitives.h"
#include "Math/Matrix4x4.h"
#include "Math/MathUtility.h"

#include "Platform/OpenGL/OpenGLDefinitions.h"
#include "Platform/OpenGL/OpenGLTexture.h"
#include "Platform/OpenGL/OpenGLRenderTarget.h"
#include "Platform/OpenGL/OpenGLAPI.h"

#include "glad/glad.h"

namespace ESource {
	
	unsigned int GetOpenGLCubemapFace(const ECubemapFace & CF) {
		switch (CF) {
			default:
			case ECubemapFace::Right:
				return GL_TEXTURE_CUBE_MAP_POSITIVE_X;
			case ECubemapFace::Left:
				return GL_TEXTURE_CUBE_MAP_NEGATIVE_X;
			case ECubemapFace::Up:
				return GL_TEXTURE_CUBE_MAP_POSITIVE_Y;
			case ECubemapFace::Down:
				return GL_TEXTURE_CUBE_MAP_NEGATIVE_Y;
			case ECubemapFace::Back:
				return GL_TEXTURE_CUBE_MAP_POSITIVE_Z;
			case ECubemapFace::Front:
				return GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
		}
	}

	OpenGLRenderTarget::OpenGLRenderTarget()
		: RenderingTextures(), Dimension(ETextureDimension::None) {
		glGenFramebuffers(1, &FramebufferObject);
		RenderbufferObject = GL_NONE;
		LOG_CORE_DEBUG(L"Render Target {} Created", FramebufferObject);
	}

	OpenGLRenderTarget::~OpenGLRenderTarget() {
		ReleaseTextures(); Unbind();
		glDeleteFramebuffers(1, &FramebufferObject);
		if (RenderbufferObject != GL_NONE)
			glDeleteRenderbuffers(1, &RenderbufferObject);
	}

	void OpenGLRenderTarget::Bind() const {
		if (!IsValid()) LOG_CORE_ERROR(L"RenderTarget is not valid");
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferObject);
	}

	void OpenGLRenderTarget::Unbind() const {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
	}

	bool OpenGLRenderTarget::CheckStatus() const {
		const GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		return Status == GL_FRAMEBUFFER_COMPLETE;
	}

	void * OpenGLRenderTarget::GetNativeObject() const {
		if (IsValid()) return (void *)(unsigned long long) FramebufferObject;
		return 0;
	}

	Texture * OpenGLRenderTarget::GetBindedTexture(int Index) const {
		if (RenderingTextures.size() < Index) return NULL;
		return RenderingTextures[Index];
	}

	void OpenGLRenderTarget::BindDepthTexture2D(Texture2D * Texture, const IntVector2 & InSize, int Lod) {
		if (!IsValid()) return;
		RenderingTextures.push_back(Texture);
		Size = InSize;
		Bind();

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, (GLuint)(unsigned long long)Texture->GetTextureObject(), Lod);
		if (RenderingTextures.size() <= 1) {
			glDrawBuffer(GL_NONE);
		}
		glReadBuffer(GL_NONE);
	}

	void OpenGLRenderTarget::BindTexture2D(Texture2D * Texture, const IntVector2 & InSize, int Lod, int TextureAttachment) {
		if (!IsValid()) return;
		RenderingTextures.push_back(Texture);
		Size = IntVector3(InSize.x, InSize.y, 1);
		Bind();

		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + TextureAttachment, (GLuint)(unsigned long long)Texture->GetTextureObject(), Lod);
		// Set the list of draw buffers.
		GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 + (GLenum)TextureAttachment };
		glDrawBuffers(1, DrawBuffers);
	}

	void OpenGLRenderTarget::BindTextures2D(Texture2D ** Textures, const IntVector2 & InSize, int * Lods, int * TextureAttachments, unsigned int Count) {
		if (!IsValid()) return;
		Size = IntVector3(InSize.x, InSize.y, 1);
		Bind();

		// Set the list of draw buffers.
		TArray<GLenum> DrawBuffers = TArray<GLenum>(Count);
		for (unsigned int i = 0; i < Count; i++) {
			RenderingTextures.push_back(Textures[i]);
			DrawBuffers[i] = GL_COLOR_ATTACHMENT0 + (GLenum)TextureAttachments[i];
			glFramebufferTexture2D(GL_FRAMEBUFFER, DrawBuffers[i], GL_TEXTURE_2D, (GLuint)(unsigned long long)Textures[i]->GetTextureObject(), Lods[i]);
		}

		glDrawBuffers(Count, &DrawBuffers[0]);

		Unbind();
	}

	void OpenGLRenderTarget::BindCubemapFace(Cubemap * Texture, const int & InSize, ECubemapFace Face, int Lod, int TextureAttachment) {
		if (!IsValid()) return;
		RenderingTextures.push_back(Texture);
		Size = IntVector3(InSize, InSize, 1);
		Bind();
		
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + TextureAttachment,
			GetOpenGLCubemapFace(Face), (GLuint)(unsigned long long)Texture->GetTextureObject(), Lod);
	}

	void OpenGLRenderTarget::CreateRenderDepthBuffer2D(EPixelFormat Format, const IntVector2 & Size) {
		Bind();
		if (RenderbufferObject == GL_NONE) {
			glGenRenderbuffers(1, &RenderbufferObject);
			glBindRenderbuffer(GL_RENDERBUFFER, RenderbufferObject);
			glRenderbufferStorage(GL_RENDERBUFFER, OpenGLPixelFormatInfo[Format].InputFormat, Size.x, Size.y);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RenderbufferObject);
		}
		Unbind();
	}

	void OpenGLRenderTarget::TransferDepthTo(RenderTarget * Target, const EPixelFormat & Value, const EFilterMode & FilterMode, const Box2D & From, const Box2D & To) {
		GLuint FramebufferTarget = Target == NULL ? 0 : ((OpenGLRenderTarget *)(Target))->FramebufferObject;
		glBindFramebuffer(GL_READ_FRAMEBUFFER, FramebufferObject);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, FramebufferTarget);
		glBlitFramebuffer((int)From.xMin, (int)From.yMin, (int)From.xMax, (int)From.yMax, (int)To.xMin, (int)To.yMin, (int)To.xMax, (int)To.yMax,
			GL_DEPTH_BUFFER_BIT, OpenGLAPI::FilterModeToBaseType(FilterMode));
	}

	void OpenGLRenderTarget::ReleaseTextures() {
		RenderingTextures.clear();
	}

	void OpenGLRenderTarget::Clear() const {
		Rendering::ClearCurrentRender(true, 0, true, 1, false, 0);
	}

	bool OpenGLRenderTarget::IsValid() const {
		return FramebufferObject != GL_FALSE;
	}

}