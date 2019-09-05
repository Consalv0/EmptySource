#pragma once

#include "CoreTypes.h"
#include "Core/Layer.h"

namespace EmptySource {

	class OObject;

	// FIX object deletion, for now deleting objects in gameloop are unknown behaviour
	class SpaceLayer : public Layer {
	public:
		static SpaceLayer * SpaceLayer::GetSpace(const size_t & Identifier);

		SpaceLayer() = delete;

		SpaceLayer(const WString & Name, unsigned int Level);

		// SpaceLayer(SpaceLayer& OtherSpace);

		virtual void OnAttach() override;

		virtual void OnAwake() override;

		virtual void OnDetach() override;

		virtual void OnRender() override;

		virtual void OnUpdate(Timestamp Stamp) override;

		virtual void OnImGuiRender() override;

		virtual void OnWindowEvent(WindowEvent& WinEvent) override;

		virtual void OnInputEvent(InputEvent& InEvent) override;

		~SpaceLayer();

		//* Destroys all objects in this Space
		void DeleteAllObjects();

		//* Destroy specific object in this Space
		void DeleteObject(OObject* object);

		WString GetFriendlyName() const;

		OObject * GetObjectByID(size_t Identifier);

		//* Get the object of this type
		template<typename T>
		T * GetFirstObject();

		//* Get the object of this type
		template<typename T>
		void GetAllObjects(TArray<T *> & List);

		//* Creates an object in this space
		template<typename T>
		T * CreateObject();

		//* Creates an object with name in this space
		template<typename T, typename... Rest>
		T * CreateObject(Rest... Args);

	protected:
		// Dictionary of all Spaces created
		static TDictionary<size_t, SpaceLayer*> AllSpaces;

	private:
		bool bAttached;

		WString Name;

		//* Dictionary that contains all the Objects in this Space
		TDictionary<size_t, OObject*> ObjectsIn;

		//* Add object in this space
		void AddObject(OObject* Object);
	};

	template<typename T>
	T * SpaceLayer::GetFirstObject() {
		for (auto & Iterator : ObjectsIn) {
			if (Iterator.second->GetObjectName() == T::GetStaticObjectName())
				return dynamic_cast<T *>(Iterator.second);
		}
		return NULL;
	}

	template<typename T>
	void SpaceLayer::GetAllObjects(TArray<T*>& List) {
		for (auto & Iterator : ObjectsIn) {
			if (Iterator.second->GetObjectName() == T::GetStaticObjectName())
				List.push_back(dynamic_cast<T *>(Iterator.second));
		}
	}

	template<typename T>
	T * SpaceLayer::CreateObject() {
		T* NewObject = new T();
		AddObject(NewObject);
		return NewObject;
	}

	template<typename T, typename ...Rest>
	T * SpaceLayer::CreateObject(Rest ...Args) {
		T* NewObject = new T(Args...);
		AddObject(NewObject);
		return NewObject;
	}

}