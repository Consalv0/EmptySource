#pragma once

#include "Components/Component.h"
#include "Events/Observer.h"

namespace ESource {

	class CRenderable : public CComponent {
		IMPLEMENT_COMPONENT(CRenderable)
	public:
		uint8_t CullingMask;

		bool bGPUInstance;

		virtual void SetMesh(RMeshPtr Value);

		virtual void SetMaterials(TArray<MaterialPtr> & Materials);

		virtual void SetMaterialAt(uint32_t At, MaterialPtr Mat);

		virtual const TDictionary<int, MaterialPtr> & GetMaterials() const;

		virtual RMeshPtr GetMesh() const;

		virtual void OnRender() override;

	protected:
		typedef CComponent Supper;
		CRenderable(GGameObject & GameObject);

		virtual void OnDelete() override;

		RMeshPtr ActiveMesh;

		TDictionary<int, MaterialPtr> Materials;
	};

}