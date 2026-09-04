#version 330 core

in vec2 vUV;
uniform sampler2D uAccumTexture;
uniform float uGlowDensity;

out vec4 FragColor;

vec4 sample_accum_2x2(vec2 uv)
{
    vec2 texel = 1.0 / vec2(textureSize(uAccumTexture, 0));
    vec2 h = 0.5 * texel;

    return 0.25 * (
        texture(uAccumTexture, uv + vec2(-h.x, -h.y)) +
        texture(uAccumTexture, uv + vec2( h.x, -h.y)) +
        texture(uAccumTexture, uv + vec2(-h.x,  h.y)) +
        texture(uAccumTexture, uv + vec2( h.x,  h.y))
    );
}

void main()
{
    vec4 accum = sample_accum_2x2(vUV);
    if (accum.a <= 1.0e-6) {
        FragColor = vec4(0.0);
        return;
    }

    vec3 meanColor = accum.rgb / accum.a;

    // Same smooth density response used by the base composite. Density is
    // then shaped before blur: values < 1 make isolated particles glow too,
    // values > 1 favor dense clusters.
    float density = 1.0 - exp(-1.72 * accum.a);
    float emission = pow(clamp(density, 0.0, 1.0), max(uGlowDensity, 0.05));

    FragColor = vec4(meanColor * emission, 1.0);
}
