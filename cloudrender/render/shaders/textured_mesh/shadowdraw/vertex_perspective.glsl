#version 330 core
#define SHADOWMAPS_MAX 6

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec2 vertexTexUV;
layout(location = 2) in vec3 vertexNorm;

// Output data ; will be interpolated for each fragment.
out VS_OUT {
	vec3 pose;
	float depth;
	vec3 normal;
	vec3 MVnormal;
	vec4 pose_shadow[SHADOWMAPS_MAX];
	vec2 texUV;
} vs_out;

// Values that stay constant for the whole mesh.
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform bool shadowmap_enabled[SHADOWMAPS_MAX];
uniform mat4 shadowmap_MVP[SHADOWMAPS_MAX];
void main(){
	mat4 MV = V*M;
	vec4 vertexPosMV = MV * vec4(vertexPos, 1);
	gl_Position = P * vertexPosMV;
	vs_out.pose = vec3(M * vec4(vertexPos, 1.0));
	vs_out.normal = mat3(M) * vertexNorm;
//	vs_out.MVnormal = -mat3(MV) * vertexNorm;
	vs_out.depth = abs(vertexPosMV.z);
	for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
		vs_out.pose_shadow[i] = shadowmap_MVP[i] * vec4(vertexPos, 1);
	vs_out.texUV = vertexTexUV;
}