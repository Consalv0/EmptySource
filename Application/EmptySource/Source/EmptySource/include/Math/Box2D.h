#pragma once

struct Box2D {
public: 
	union {
		struct { float Left, Bottom, Right, Top; };
		struct { float MinX, MinY, MaxX, MaxY; };
	};
};