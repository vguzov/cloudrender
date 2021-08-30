#version 330 core
#define SHADOWMAPS_MAX 6

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec4 vertexColor;
layout(location = 2) in int vertexId;
//layout(location = 3) in int vertexNorm;

out VS_OUT {
    vec4 color;
	int inst_id;
	float depth;
//	vec3 normal;
	vec4 pose_shadow[SHADOWMAPS_MAX];
} vs_out;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform bool shadowmap_enabled[SHADOWMAPS_MAX];
uniform mat4 shadowmap_MVP[SHADOWMAPS_MAX];
void main(){
	mat4 MV = V*M;
	vec4 vertexPosMV = MV * vec4(vertexPos, 1);
	gl_Position = P * vertexPosMV;

	vs_out.color = vertexColor;
	vs_out.inst_id = vertexId;
//	vs_out.normal = mat3(M) * vertexNorm;
	vs_out.depth = abs(vertexPosMV.z);
	for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
		vs_out.pose_shadow[i] = shadowmap_MVP[i] * vec4(vertexPos, 1);
}