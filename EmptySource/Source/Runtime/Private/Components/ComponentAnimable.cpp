
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "Core/CoreTime.h"
#include "Core/Application.h"

#include "Components/ComponentAnimable.h"

namespace ESource {

	CAnimable::CAnimable(GGameObject & GameObject)
		: CComponent(L"Animable", GameObject), Track(NULL), CurrentAnimationTime(0.F), AnimationSpeed(1.F), EventsCallback(), bLoop(false), bPlaying(true) {
	}

	void CAnimable::OnUpdate(const Timestamp& Stamp) {
		if (Track && bPlaying) {
			CurrentAnimationTime += Stamp.GetDeltaTime<Time::Second>() * AnimationSpeed * Track->TicksPerSecond;
			if (CurrentAnimationTime > Track->Duration) {
				CallEventsOnEndAnimation();
				if (bLoop)
					CurrentAnimationTime = std::fmod(CurrentAnimationTime, Track->Duration);
				else {
					CurrentAnimationTime = 0.F;
					bPlaying = false;
				}
			}
			UpdateHierarchy();
		}
	}

	void CAnimable::AddEventOnEndAnimation(const NString & Name, const CallbackFunctionPointer & Function) {
		EventsCallback.insert_or_assign(Name, Function);
	}

	bool CAnimable::Initialize() {
		return true;
	}

	void CAnimable::OnDelete() {
	}

	void CAnimable::CallEventsOnEndAnimation() {
		for (auto & Event : EventsCallback) {
			Event.second();
		}
	}

	size_t FindPosition(double AnimationTime, const AnimationTrackNode * NodeAnim) {
		for (size_t i = 0; i < NodeAnim->Positions.size() - 1; i++) {
			if (AnimationTime < NodeAnim->Positions[i + 1].Time)
				return i;
		}

		return NodeAnim->Positions.size() - 1;
	}


	size_t FindRotation(double AnimationTime, const AnimationTrackNode * NodeAnim) {
		ES_CORE_ASSERT(!NodeAnim->Rotations.empty());

		for (size_t i = 0; i < NodeAnim->Rotations.size() - 1; i++) {
			if (AnimationTime < NodeAnim->Rotations[i + 1].Time)
				return i;
		}

		return NodeAnim->Rotations.size() - 1;
	}


	size_t FindScaling(double AnimationTime, const AnimationTrackNode * NodeAnim) {
		ES_CORE_ASSERT(!NodeAnim->Scalings.empty());

		for (size_t i = 0; i < NodeAnim->Scalings.size() - 1; i++) {
			if (AnimationTime < NodeAnim->Scalings[i + 1].Time)
				return i;
		}

		return  NodeAnim->Scalings.size() - 1;
	}


	Vector3 InterpolateTranslation(double AnimationTime, const AnimationTrackNode * NodeAnim) {
		if (NodeAnim->Positions.size() == 1) {
			// No interpolation necessary for single value
			return NodeAnim->Positions[0];
		}

		size_t PositionIndex = FindPosition(AnimationTime, NodeAnim);
		size_t NextPositionIndex = (PositionIndex + 1);
		ES_CORE_ASSERT(NextPositionIndex < NodeAnim->Positions.size());
		double DeltaTime = (NodeAnim->Positions[NextPositionIndex].Time - NodeAnim->Positions[PositionIndex].Time);
		double Factor = (AnimationTime - NodeAnim->Positions[PositionIndex].Time) / DeltaTime;
		// if (Factor <= 1.F) LOG_CORE_ERROR("Factor must be below 1.0");
		Factor = Math::Clamp01(Factor);
		const Vector3& Start = NodeAnim->Positions[PositionIndex];
		const Vector3& End = NodeAnim->Positions[NextPositionIndex];
		return Math::Mix(Start, End, (float)Factor);
	}


	Quaternion InterpolateRotation(double AnimationTime, const AnimationTrackNode * NodeAnim) {
		if (NodeAnim->Rotations.size() == 1) {
			// No interpolation necessary for single value
			return NodeAnim->Rotations[0];
		}

		size_t RotationIndex = FindRotation(AnimationTime, NodeAnim);
		size_t NextRotationIndex = (RotationIndex + 1);
		ES_CORE_ASSERT(NextRotationIndex < NodeAnim->Rotations.size());
		double DeltaTime = (NodeAnim->Rotations[NextRotationIndex].Time - NodeAnim->Rotations[RotationIndex].Time);
		double Factor = (AnimationTime - NodeAnim->Rotations[RotationIndex].Time) / DeltaTime;
		// if (Factor <= 1.F) LOG_CORE_ERROR("Factor must be below 1.0");
		Factor = Math::Clamp01(Factor);
		const Quaternion& StartRotationQ = NodeAnim->Rotations[RotationIndex];
		const Quaternion& EndRotationQ = NodeAnim->Rotations[NextRotationIndex];
		auto Quat = Quaternion();
		Quaternion::Interpolate(Quat, StartRotationQ, EndRotationQ, (float)Factor);
		Quat.Normalize();
		return Quat;
	}


	Vector3 InterpolateScale(double AnimationTime, const AnimationTrackNode * NodeAnim) {
		if (NodeAnim->Scalings.size() == 1) {
			// No interpolation necessary for single value
			return NodeAnim->Scalings[0];
		}

		size_t Index = FindScaling(AnimationTime, NodeAnim);
		size_t NextIndex = (Index + 1);
		ES_CORE_ASSERT(NextIndex < NodeAnim->Scalings.size());
		double DeltaTime = (NodeAnim->Scalings[NextIndex].Time - NodeAnim->Scalings[Index].Time);
		double Factor = (AnimationTime - NodeAnim->Scalings[Index].Time) / DeltaTime;
		// if (Factor <= 1.F) LOG_CORE_ERROR("Factor must be below 1.0");
		Factor = Math::Clamp01(Factor);
		const Vector3& Start = NodeAnim->Scalings[Index];
		const Vector3& End = NodeAnim->Scalings[NextIndex];
		auto Delta = End - Start;
		return Start + (float)Factor * Delta;
	}

	void CAnimable::UpdateHierarchy() {
		if (Track) {
			for (auto & AnimationNode : Track->AnimationNodes) {
				GGameObject * Find = FindInHiererchy(&GetGameObject(), AnimationNode.Name);
				if (Find != NULL) {
					Find->LocalTransform.Position = InterpolateTranslation(CurrentAnimationTime, &AnimationNode);
					Find->LocalTransform.Rotation = InterpolateRotation(CurrentAnimationTime, &AnimationNode);
					Find->LocalTransform.Scale    = InterpolateScale(CurrentAnimationTime, &AnimationNode);
				}
			}
		}
	}

	GGameObject * CAnimable::FindInHiererchy(GGameObject * GO, const IName & Name) {
		if (GO->GetName() == Name) { return GO; }
		TArray<GGameObject *> Children;
		GO->GetAllChildren<GGameObject>(Children);
		for (auto & Child : Children) {
			GGameObject * Find = FindInHiererchy(Child, Name);
			if (Find != NULL)
				return Find;
		}
		return NULL;
	}

}