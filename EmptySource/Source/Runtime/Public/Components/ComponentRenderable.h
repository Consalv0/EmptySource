#pragma once

#include "Components/Component.h"
#include "Events/Observer.h"

namespace EmptySource {

	class CRenderable : public CComponent {
	protected:
		typedef CComponent Supper;
		friend class GGameObject;
		friend class Space;
		CRenderable(GGameObject & GameObject);

		virtual bool Initialize();

		virtual void OnDelete();

		virtual void SetMesh(MeshPtr Value);

		virtual void SetMaterials(TArray<class Material *> Materials);

		virtual void SetMaterialAt(unsigned int At, class Material * Mat);

		MeshPtr Model;

	public:

		virtual void OnRender();
	};

}