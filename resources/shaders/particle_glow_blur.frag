#version 330 core

in vec2 vUV;
uniform sampler2D uInputTexture;
uniform vec2 uDirection;
uniform float uRadius;

out vec4 FragColor;

void main()
{
    vec2 texel = 1.0 / vec2(textureSize(uInputTexture, 0));
    vec2 stepUV = texel * uDirection * max(uRadius, 0.0);

    // Five texture reads approximate a nine-tap Gaussian by relying on linear
    // filtering between adjacent texels. This is deliberately small and fast.
    vec3 color = texture(uInputTexture, vUV).rgb * 0.2270270270;
    color += texture(uInputTexture, vUV + stepUV * 1.3846153846).rgb * 0.3162162162;
    color += texture(uInputTexture, vUV - stepUV * 1.3846153846).rgb * 0.3162162162;
    color += texture(uInputTexture, vUV + stepUV * 3.2307692308).rgb * 0.0702702703;
    color += texture(uInputTexture, vUV - stepUV * 3.2307692308).rgb * 0.0702702703;

    FragColor = vec4(color, 1.0);
}
