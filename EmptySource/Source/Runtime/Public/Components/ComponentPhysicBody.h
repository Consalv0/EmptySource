#pragma once

#include "Components/Component.h"

namespace ESource {

	class CPhysicBody : public CComponent {
		IMPLEMENT_COMPONENT(CPhysicBody)
	public:
		void SetMesh(RMeshPtr & Mesh);

		struct MeshData * GetMeshData();

		bool bDoubleSided;

	protected:
		RMeshPtr ActiveMesh;
		
		typedef CComponent Supper;
		CPhysicBody(GGameObject & GameObject);

		virtual void OnRender() override;

		virtual void OnAttach() override;

		virtual void OnDelete() override;
	};

}