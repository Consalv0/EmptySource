#pragma once

#include "Core/SpaceLayer.h"
#include "Core/Name.h"

#define IMPLEMENT_OBJECT(Name) protected: \
inline virtual WString GetObjectName() override { return L#Name;} \
static WString GetStaticObjectName() { return L#Name;} \
friend class SpaceLayer; 

namespace EmptySource {

	class OObject {
	public:
		inline virtual WString GetObjectName() { return L"OObject"; }

		static WString GetStaticObjectName() { return L"OObject"; }

		// Safe methos to delete this object removing it from the Space
		virtual void OnDelete() {};

		// Name of this object
		inline const IName & GetName() const { return Name; };

		inline bool IsAttached() { return bAttached; }

	protected:
		friend class SpaceLayer;

		SpaceLayer * SpaceIn;

		IName Name;

		OObject();

		OObject(const IName& Name);

		virtual void OnAttach() { bAttached = true; };

		virtual void OnAwake() {};

		virtual void OnDetach() {};

	private:
		bool bAttached;
	};

}