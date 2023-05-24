#version 330 core

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec4 vertexColor;
layout(location = 2) in vec3 vertexNorm;

out VS_OUT {
    vec4 color;
	vec3 norm;
	vec4 poseMV;
} vs_out;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
void main(){
	mat4 MV = V*M;
	vec4 vertexPosMV = MV * vec4(vertexPos, 1);
	gl_Position = P * vertexPosMV;
	vs_out.color = vertexColor;
	vs_out.poseMV = vertexPosMV;
	vs_out.norm = vertexNorm;
}