#pragma once

#include "Core/Object.h"

namespace EmptySource {

	class GGameObject;

	class CComponent : public OObject {
	protected:
		typedef OObject Supper;
		friend class GGameObject;
		friend class Space;

		CComponent(GGameObject & GameObject);
		CComponent(WString Name, GGameObject & GameObject);

		virtual void OnDelete();

		virtual bool Initialize();

		GGameObject & Holder;

	public:

		GGameObject & GetGameObject() const;

	};

}