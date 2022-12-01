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

void main() {
	color = vcolor;
	for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i) {
		bool is_shadow = shadow_calculation(shadowmaps[i], pose_shadow[i]);
		color = is_shadow ? vec4(shadow_color.rgb*shadow_color.a+color.rgb*(1-shadow_color.a), shadow_color.a + color.a*(1-shadow_color.a)):color;
	}
}