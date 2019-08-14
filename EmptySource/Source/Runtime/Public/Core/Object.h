#pragma once

#include "Core/Space.h"
#include "Core/IIdentifier.h"

namespace EmptySource {

	class OObject : public IIdentifier {
	protected:
		friend class Space;

		Space* SpaceIn;

		WString Name;

		OObject();
		OObject(const WString& Name);

		// Space initializer
		virtual bool Initialize() { return true; };

	public:

		// Safe methos to delete this object removing it from the Space
		virtual void OnDelete() {};

		// Name of this object
		WString GetName();
	};

}