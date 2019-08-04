#pragma once

#include "CoreTypes.h"
#include "IIdentifier.h"

namespace EmptySource {

	class OObject;

	class Space : public IIdentifier {
	public:
		//* Get active main Space
		static Space* GetMainSpace();

		//* Get Space by IIdentifier
		static Space* GetSpace(const size_t& Identifier);

		static Space* CreateSpace(const WString& Name);

		static void Destroy(Space * OtherSpace);

		//* Destroys all objects in this Space
		void DeleteAllObjects();

		//* Destroy specific object in this Space
		void DeleteObject(OObject* object);

		WString GetFriendlyName() const;

		//* Creates an object in this space
		template<typename T>
		T * CreateObject();

		//* Creates an object with name in this space
		template<typename T, typename... Rest>
		T * CreateObject(Rest... Args);

	private:
		Space();

		Space(const WString & Name);

		Space(Space& OtherSpace);

		WString Name;

		//* Dictionary that contains all the Objects in this Space
		TDictionary<size_t, OObject*> ObjectsIn;

		//* Add object in this space
		void AddObject(OObject* Object);

	protected:

		// Dictionary of all Spaces created
		static TDictionary<size_t, Space*> AllSpaces;
	};

	template<typename T>
	T * Space::CreateObject() {
		T* NewObject = new T();
		AddObject(NewObject);
		return NewObject;
	}

	template<typename T, typename ...Rest>
	T * Space::CreateObject(Rest ...Args) {
		T* NewObject = new T(Args...);
		AddObject(NewObject);
		return NewObject;
	}

}