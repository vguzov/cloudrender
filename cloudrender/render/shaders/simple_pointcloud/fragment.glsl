#version 330 core

in vec4 vcolor;
flat in int frag_inst_id;

layout(location = 0) out vec4 color;
layout(location = 1) out int inst_id;

uniform vec4 overlay_color;
uniform vec3 hsv_multiplier;

vec4 alpha_blending(vec4 orig_color, vec4 overlay_color)
{
    float res_alpha = overlay_color.a + orig_color.a*(1-overlay_color.a);
	return vec4((res_alpha==0.0)?orig_color.rgb:((overlay_color.rgb*overlay_color.a+orig_color.rgb*(1-overlay_color.a))/res_alpha), res_alpha);
}

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
	color = vcolor;
	color = vec4(hsv2rgb(rgb2hsv(color.rgb)*hsv_multiplier), color.a);
	color = alpha_blending(color, overlay_color);
	inst_id = frag_inst_id;
}