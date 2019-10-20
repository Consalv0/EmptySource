#pragma once

#include "Core/Object.h"
#include "Core/Transform.h"
#include "Components/Component.h"

namespace ESource {

	//* Basic class for any object that contains components and a spatial representation
	class GGameObject : public OObject {
		IMPLEMENT_OBJECT(GGameObject)
	public:
		Transform LocalTransform;

		void AttachTo(GGameObject * Parent);

		void DeatachFromParent();

		bool Contains(GGameObject * Other) const;

		bool ContainsRecursiveUp(GGameObject * Other) const;

		bool ContainsRecursiveDown(GGameObject * Other) const;

		virtual void DestroyComponent(CComponent * Component);

		inline bool IsRoot() const { return Parent == NULL; };

		Matrix4x4 GetWorldMatrix() const;

		Transform GetWorldTransform() const;

		template<typename T>
		T * GetFirstComponent();

		template<typename T>
		void GetAllComponents(TArray<T *> & Components);

		template<typename T, typename... Rest>
		T * CreateComponent(Rest... Args);

		template<typename T>
		void GetAllChildren(TArray<T *> & OutChildren);

	private:
		TArray<GGameObject *> Children;

		GGameObject * Parent;

		typedef OObject Supper;

		GGameObject();
		GGameObject(const WString & Name);
		GGameObject(const Transform & LocalTransform);
		GGameObject(const WString & Name, const Transform & LocalTransform);

		//* Dictionary that contains all the Components in this GameObject
		TDictionary<size_t, CComponent*> ComponentsIn;

		//* Dictionary that contains all the Components in this GameObject to destroy
		TDictionary<size_t, CComponent*> ComponentsOut;

		//* Add component
		void AttachComponent(CComponent * Component);

		void DetachComponent(CComponent * Component);

		void DeleteOutComponents();

		void DeatachAllComponents();

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
			if (Iterator.second && Iterator.second->GetObjectName() == T::GetStaticObjectName())
				return dynamic_cast<T *>(Iterator.second);
		}

		return NULL;
	}

	template<typename T>
	void GGameObject::GetAllComponents(TArray<T *>& Components) {
		for (auto & Iterator : ComponentsIn) {
			if (Iterator.second && Iterator.second->GetObjectName() == T::GetStaticObjectName())
				Components.push_back(dynamic_cast<T *>(Iterator.second));
		}
	}

	template<typename T, typename ...Rest>
	T * GGameObject::CreateComponent(Rest ...Args) {
		T* NewObject = new T(*this, Args...);
		AttachComponent(NewObject);
		return NewObject;
	}

	template<typename T>
	void GGameObject::GetAllChildren(TArray<T *>& OutChildren) {
		for (auto & Iterator : Children) {
			if (Iterator->GetObjectName() == T::GetStaticObjectName())
				OutChildren.push_back(dynamic_cast<T *>(Iterator));
		}
	}

}