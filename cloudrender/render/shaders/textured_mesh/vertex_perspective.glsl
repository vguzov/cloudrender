#version 330 core

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
	vec2 texUV;
} vs_out;

// Values that stay constant for the whole mesh.
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
void main(){
	mat4 MV = V*M;
	vec4 vertexPosMV = MV * vec4(vertexPos, 1);
	gl_Position = P * vertexPosMV;
	vs_out.pose = vec3(M * vec4(vertexPos, 1.0));
	vs_out.normal = mat3(M) * vertexNorm;
//	vs_out.MVnormal = -mat3(MV) * vertexNorm;
	vs_out.depth = abs(vertexPosMV.z);
	vs_out.texUV = vertexTexUV;
}