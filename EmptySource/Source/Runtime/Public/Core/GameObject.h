#pragma once

#include "Core/Object.h"
#include "Core/Transform.h"
#include "Components/Component.h"

namespace EmptySource {

	//* Basic class for any object that contains components and a spacial representation
	class GGameObject : public OObject {
		IMPLEMENT_OBJECT(GGameObject)
	public:
		Transform Transformation;

		TArray<GGameObject> Children;

		template<typename T>
		T * GetFirstComponent();

		template<typename T, typename... Rest>
		T * CreateComponent(Rest... Args);

	private:
		typedef OObject Supper;

		GGameObject();
		GGameObject(const WString & Name);
		GGameObject(const Transform & LocalTransform);
		GGameObject(const WString & Name, const Transform & LocalTransform);

		//* Dictionary that contains all the Components in this GameObject
		TDictionary<size_t, CComponent*> ComponentsIn;

		//* Add component
		void AddComponent(CComponent * Component);

		void DeleteComponent(CComponent * Component);
		void DeleteAllComponents();

		virtual void OnRender();

		virtual void OnUpdate(const Timestamp& Stamp) ;

		virtual void OnImGuiRender();

		virtual void OnWindowEvent(WindowEvent& WinEvent);

		virtual void OnInputEvent(InputEvent& InEvent);

		virtual void OnDelete();
	};

	template<typename T>
	T * GGameObject::GetFirstComponent() {
		for (auto & Iterator : ComponentsIn) {
			if (Iterator.second->GetObjectName() == T::GetStaticObjectName())
				return dynamic_cast<T *>(Iterator.second);
		}

		return NULL;
	}

	template<typename T, typename ...Rest>
	T * GGameObject::CreateComponent(Rest ...Args) {
		T* NewObject = new T(*this, Args...);
		AddComponent(NewObject);
		return NewObject;
	}

}