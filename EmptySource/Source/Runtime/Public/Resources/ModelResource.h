#pragma once

#include "Resources/MaterialManager.h"
#include "Resources/ResourceHolder.h"
#include "Resources/MeshResource.h"
#include "Core/Transform.h"

#include "Rendering/Animation.h"

namespace ESource {

	struct ModelNode {
		NString Name;
		Transform LocalTransform;
		bool bHasMesh;
		size_t MeshKey;
		ModelNode * Parent;
		TArray<ModelNode *> Children;

		ModelNode(const NString& Name) : Name(Name), LocalTransform(), bHasMesh(false), MeshKey(0), Parent(NULL), Children() {};

		ModelNode & operator=(const ModelNode & Other) {
			Name = Other.Name;
			LocalTransform = Other.LocalTransform;
			bHasMesh = Other.bHasMesh;
			MeshKey = Other.MeshKey;
			Parent = NULL;
			Children.clear();
			for (auto & OtherChild : Other.Children) {
				ModelNode * Child = new ModelNode(Name);
				*Child = *OtherChild;
				Child->Parent = this;
				Children.push_back(Child);
			}
			return *this;
		}

		~ModelNode() {
			for (auto & Child : Children) {
				delete Child;
			}
		}

		inline ModelNode * AddChild(const NString& Name) {
			ModelNode * Child = new ModelNode(Name);
			Child->Parent = this;
			Children.push_back(Child);
			return Child;
		}
	};

	typedef std::shared_ptr<class RModel> RModelPtr;

	class RModel : public ResourceHolder {
	public:
		bool bOptimizeOnLoad;

		~RModel();

		virtual bool IsValid() const override;

		virtual void Load() override;

		virtual void LoadAsync() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }

		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Model; }

		virtual inline size_t GetMemorySize() const override;

		static inline EResourceType GetType() { return EResourceType::RT_Model; };

		ModelNode * GetHierarchyParentNode() { return &ParentNode; }

		TArray<ModelNode *> GetTraversalNodes(const std::function<bool(ModelNode *&)> & ComparisionFunction);

		const TDictionary<size_t, RMeshPtr> & GetMeshes() const { return Meshes; };

		const TArray<AnimationTrack> & GetAnimations() const { return Animations; };

	protected:
		friend class ModelManager;

		RModel(const IName & Name, const WString & Origin, bool bOptimize = false);

	private:
		void GetTraversalNodes(ModelNode * Node, TArray<ModelNode*>& Vector, const std::function<bool(ModelNode *&)> & ComparisionFunction);

		TDictionary<size_t, RMeshPtr> Meshes;

		TDictionary<NString, Material> DefaultMaterials;

		TArray<AnimationTrack> Animations;

		ModelNode ParentNode;
	};

}