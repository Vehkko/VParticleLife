#version 330 core

in vec2 vUV;
uniform sampler2D uAccumTexture;
uniform sampler2D uGlowTexture;
uniform int uGlowEnabled;
uniform float uGlowStrength;
uniform float uGlowExposure;

out vec4 FragColor;

void main()
{
    vec4 accum = texture(uAccumTexture, vUV);

    vec3 baseColor = vec3(0.0);
    if (accum.a > 1.0e-6) {
        // Order-independent local mean color. Physical GPU reordering can no
        // longer change the mixed color of overlapping particle types.
        vec3 meanColor = accum.rgb / accum.a;
        float opacity = 1.0 - exp(-1.72 * accum.a);
        baseColor = meanColor * opacity;
    }

    if (uGlowEnabled != 0) {
        vec3 bloom = texture(uGlowTexture, vUV).rgb * max(uGlowStrength, 0.0);

        // Tone-map only the bloom contribution, then combine it with the base
        // using a bounded screen operation. This preserves the existing base
        // rendering when glow is zero and avoids additive white clipping.
        vec3 glowColor = 1.0 - exp(-bloom * max(uGlowExposure, 0.0));
        baseColor = 1.0 - (1.0 - clamp(baseColor, 0.0, 1.0)) * (1.0 - glowColor);
    }

    FragColor = vec4(baseColor, 1.0);
}
