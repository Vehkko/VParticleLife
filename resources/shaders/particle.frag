#version 330 core

flat in vec4 vColor;
in float vPointAlphaScale;

out vec4 FragColor;

void main()
{
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r = length(p);

    if (r > 1.0)
        discard;

    float core = 1.0 - smoothstep(0.45, 0.60, r);
    float glow = 1.0 - smoothstep(0.35, 1.0, r);

    float weight = vColor.a * vPointAlphaScale *
                   min(1.0, core + 0.18 * glow);

    // The framebuffer uses additive accumulation. Store premultiplied color
    // plus total coverage; the second pass normalizes by total coverage.
    FragColor = vec4(vColor.rgb * weight, weight);
}
