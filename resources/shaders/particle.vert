#version 330 core

layout(location = 0) in float aX;
layout(location = 1) in float aY;

uniform vec2 uHalfSize;

void main()
{
    gl_Position = vec4(aX / uHalfSize.x, aY / uHalfSize.y, 0.0, 1.0);
    gl_PointSize = 6.0;
}
