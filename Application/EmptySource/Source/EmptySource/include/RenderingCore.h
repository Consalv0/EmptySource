#pragma once

enum DepthFunctionMode {
	Never			= 0x0200,
	Less			= 0x0201,
	Equal			= 0x0202,
	LessEqual		= 0x0203,
	Greater			= 0x0204,
	NotEqual		= 0x0205,
	GreaterEqual	= 0x0206,
	Always			= 0x0207
};

enum CullMode {
	None       = 0,
	FrontLeft  = 0x0400,
	FrontRight = 0x0401,
	BackLeft   = 0x0402,
	BackRight  = 0x0403,
	Front      = 0x0404,
	Back       = 0x0405,
	Left       = 0x0406,
	Right      = 0x0407,
	FrontBack  = 0x0408,
};

enum RenderMode {
	Point = 0x1B00,
	Line  = 0x1B01,
	Fill  = 0x1B02,
};