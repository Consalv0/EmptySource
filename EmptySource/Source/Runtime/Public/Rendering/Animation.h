#pragma once

namespace ESource {

	struct KeyFrame {
		class AnimationTrackNode * ParentNode;
		double Time;

		KeyFrame(AnimationTrackNode & Node, const double & Time) : ParentNode(&Node), Time(Time) {};
	};

	struct Vector3Key : KeyFrame, Vector3 { 
		Vector3Key(AnimationTrackNode & Node, double Time, float x, float y, float z) : KeyFrame(Node, Time), Vector3(x, y, z) {};
	};
	struct QuaternionKey : KeyFrame, Quaternion {
		QuaternionKey(AnimationTrackNode & Node, double Time, float w, float x, float y, float z) : KeyFrame(Node, Time), Quaternion(w, x, y, z) {};
	};

	class AnimationTrackNode {
	public:
		class AnimationTrack * ParentTrack;
		IName Name;

		TArray<Vector3Key> Positions;
		TArray<QuaternionKey> Rotations;
		TArray<Vector3Key> Scalings;

		int NodeLevel;

		AnimationTrackNode(AnimationTrack & Track, const IName & Name) 
			: ParentTrack(&Track), Name(Name), NodeLevel(0), Positions(), Rotations(), Scalings() {};
	};

	// Copy of the structure of assimp, but with the math classes of the engine. Thanks Assimp!
	class AnimationTrack {
	public:
		NString Name;

		double Duration;
		double TicksPerSecond;

		TList<AnimationTrackNode> AnimationNodes;
	};

}