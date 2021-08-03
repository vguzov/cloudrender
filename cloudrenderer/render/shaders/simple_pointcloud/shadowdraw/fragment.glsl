#version 400
#define SHADOWMAPS_MAX 6

// Interpolated values from the vertex shaders
in vec4 vcolor;
flat in int frag_inst_id;
in vec4 pose_shadow;
in vec3 vnormal;

uniform sampler2D shadowmaps[SHADOWMAPS_MAX];
uniform vec4 shadow_color;

// Ouput data
layout(location = 0) out vec4 color;

bool shadow_calculation(sampler2D shadowmap, vec4 pose_shadow)
{
	vec3 projected_shadow = pose_shadow.xyz/pose_shadow.w;
	projected_shadow = projected_shadow*0.5+0.5;
	float closest_depth = texture(shadowmap, projected_shadow.xy).r;
	float current_depth = projected_shadow.z;
	bool shadow = (projected_shadow.x>=0 && projected_shadow.x<=1 &&projected_shadow.y >=0 && projected_shadow.y<=1 &&
	projected_shadow.z<1) && (current_depth > closest_depth) && (current_depth < closest_depth+0.5);
//	bool shadow = closest_depth == 0;
	return shadow;
}

void main() {
	color = vcolor;
	for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i) {
		bool is_shadow = shadow_calculation(shadowmaps[i], pose_shadow[i]);
		color = is_shadow ? vec4(shadowColor.rgb*shadowColor.a+color.rgb*(1-shadowColor.a), shadowColor.a + color.a*(1-shadowColor.a)):color;
	}
}