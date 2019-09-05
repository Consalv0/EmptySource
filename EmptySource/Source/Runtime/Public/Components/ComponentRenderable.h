#pragma once

#include "Components/Component.h"
#include "Events/Observer.h"

namespace EmptySource {

	class CRenderable : public CComponent {
		IMPLEMENT_COMPONENT(CRenderable)
	public:
		virtual void SetMesh(MeshPtr Value);

		virtual void SetMaterials(TArray<MaterialPtr> & Materials);

		virtual void SetMaterialAt(unsigned int At, MaterialPtr Mat);

		virtual const TDictionary<int, MaterialPtr> & GetMaterials() const;

		virtual MeshPtr GetMesh() const;

		virtual void OnRender();

	protected:
		typedef CComponent Supper;
		CRenderable(GGameObject & GameObject);

		virtual bool Initialize();

		virtual void OnDelete();

		MeshPtr ActiveMesh;

		TDictionary<int, MaterialPtr> Materials;
	};

}