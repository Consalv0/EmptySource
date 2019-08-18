
#include "CoreMinimal.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/Texture.h"
#include "Rendering/GLFunctions.h"

namespace EmptySource {

	RenderTarget::RenderTarget() {
		Resolution = 0;
		TextureColor0Target = NULL;
		RenderbufferObject = 0;
		FramebufferObject = 0;
	}

	IntVector2 RenderTarget::GetDimension() const {
		return Resolution;
	}

	void RenderTarget::PrepareTexture(const Texture2DPtr Texture, const int& Lod, const int& TextureAttachment) {
		TextureColor0Target = Texture;
		if (!IsValid()) return;
		Use();

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RenderbufferObject);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + TextureAttachment, (unsigned int)(unsigned long long)TextureColor0Target->GetTextureObject(), Lod);
		// Set the list of draw buffers.
		GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 + (GLenum)TextureAttachment };
		glDrawBuffers(1, DrawBuffers);
	}

	void RenderTarget::PrepareTexture(const CubemapPtr Texture, const int& TexturePos, const int& Lod, const int& TextureAttachment) {
		TextureColor0Target = Texture;
		if (!IsValid()) return;
		Use();

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RenderbufferObject);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + TextureAttachment,
			GL_TEXTURE_CUBE_MAP_POSITIVE_X + TexturePos, (unsigned int)(unsigned long long)TextureColor0Target->GetTextureObject(), Lod);
	}

	void RenderTarget::Resize(const int & x, const int & y) {
		Use();
		Resolution = { x, y };
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, x, y);
		glViewport(0, 0, Resolution.x, Resolution.y);
	}

	void RenderTarget::SetUpBuffers() {
		if (FramebufferObject != GL_FALSE && RenderbufferObject != GL_FALSE) return;

		glGenFramebuffers(1, &FramebufferObject);
		glGenRenderbuffers(1, &RenderbufferObject);
	}

	bool RenderTarget::CheckStatus() const {
		const GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		return Status == GL_FRAMEBUFFER_COMPLETE;
	}

	void RenderTarget::Use() const {
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferObject);
		glBindRenderbuffer(GL_RENDERBUFFER, RenderbufferObject);
	}

	void RenderTarget::Clear() const {
		Use();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	bool RenderTarget::IsValid() const {
		return TextureColor0Target != NULL && TextureColor0Target->IsValid() && FramebufferObject != GL_FALSE && RenderbufferObject != GL_FALSE;
	}

	void RenderTarget::Delete() {
		glDeleteFramebuffers(1, &FramebufferObject);
		glDeleteRenderbuffers(1, &RenderbufferObject);
	}

}