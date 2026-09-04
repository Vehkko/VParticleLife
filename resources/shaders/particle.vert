#version 330 core

layout(location = 0) in float aX;
layout(location = 1) in float aY;
layout(location = 2) in uint  aType;

uniform vec2 uCenter;
uniform vec2 uHalfSize;
uniform float uPointScale;
uniform float uParticleSize[16];
uniform vec4  uParticleColor[16];

flat out vec4 vColor;
out float vPointAlphaScale;

void main()
{
    vec2 world = vec2(aX, aY) - uCenter;
    gl_Position = vec4(world / uHalfSize, 0.0, 1.0);

    uint typeIndex = min(aType, 15u);
    float pointSize = uParticleSize[typeIndex] * uPointScale;
    gl_PointSize = max(1.0, pointSize);

    vPointAlphaScale = min(1.0, pointSize * pointSize);
    vColor = uParticleColor[typeIndex];
}
