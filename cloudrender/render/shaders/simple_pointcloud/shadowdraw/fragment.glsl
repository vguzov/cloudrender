#version 400
#define SHADOWMAPS_MAX 6
#define SHADOWDEPTH_EPS 5e-3

in vec4 vcolor;
flat in int frag_inst_id;
in vec4 pose_shadow[SHADOWMAPS_MAX];
in vec3 vnormal;

uniform bool shadowmap_enabled[SHADOWMAPS_MAX];
uniform sampler2D shadowmaps[SHADOWMAPS_MAX];
uniform vec4 shadow_color;
uniform vec4 overlay_color;
uniform vec3 hsv_multiplier;


layout(location = 0) out vec4 color;

bool shadow_calculation(sampler2D shadowmap, vec4 pose_shadow)
{
	vec3 projected_shadow = pose_shadow.xyz/pose_shadow.w;
	projected_shadow = projected_shadow*0.5+0.5;
	float closest_depth = texture(shadowmap, projected_shadow.xy).r;
	float current_depth = projected_shadow.z;
	bool shadow = (projected_shadow.x>=0 && projected_shadow.x<=1 &&projected_shadow.y >=0 && projected_shadow.y<=1 &&
	projected_shadow.z<1) && (current_depth-SHADOWDEPTH_EPS > closest_depth) && (current_depth < closest_depth+0.5);
	return shadow;
}

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
	for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i) {
		bool is_shadow = shadow_calculation(shadowmaps[i], pose_shadow[i]);
		color = is_shadow ? vec4(shadow_color.rgb*shadow_color.a+color.rgb*(1-shadow_color.a), shadow_color.a + color.a*(1-shadow_color.a)):color;
		color = vec4(hsv2rgb(rgb2hsv(color.rgb)*hsv_multiplier), color.a);
		color = alpha_blending(color, overlay_color);
	}
}