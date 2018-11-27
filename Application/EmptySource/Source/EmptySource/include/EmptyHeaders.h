#pragma once

// Include windows headers
#include <windows.h>

// Include GLAD
// Library to make function loaders for OpenGL
#include <External/GLAD/include/glad.h>

// Include GLFW
// Library to make crossplataform input and window creation
#include <External/GLFW/glfw3.h>

// Include main standard headers
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

// Include standar headers
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <conio.h>
#include <memory>

#include <vector>
template<class T>
using SArray = std::vector<T>;

// Shader Locations
#define ES_LOCATION_VERTEX 0
#define ES_LOCATION_NORMAL 1
#define ES_LOCATION_TEXCOORD 2
#define ES_LOCATION_TANGENT 3
#define ES_LOCATION_JOINTIDS 4
#define ES_LOCATION_WEIGHTS 5