#ifndef _masks_h
#define _masks_h

#include "string.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define _USE_MATH_DEFINES
#ifndef M_PI
#    define M_PI 3.14159265358979323846264338327
#endif


void project(
    float* obj, 
    const int ox,  
    const int oy,  
    const int oz, 
    const float* gridx, 
    const float* gridy, 
    const float* gridz,
    const float dsrc,
    const float ddet,
    const float* detgridx,
    const float* detgridy,
    const int dx,  
    const int dy,  
    const float* srcgridx,
    const float* srcgridy,
    float* proj, 
    const int px,  
    const int py);

#endif
