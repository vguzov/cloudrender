#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec4 vertexColor;
layout(location = 2) in int vertexId;

// Output data ; will be interpolated for each fragment.
out VS_OUT {
    vec4 color;
	int inst_id;
	float depth;
} vs_out;

// Values that stay constant for the whole mesh.
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
void main(){
	mat4 MV = V*M;
	vec4 vertexPosMV = MV * vec4(vertexPos, 1);
	gl_Position = P * vertexPosMV;

	vs_out.color = vertexColor;
	vs_out.inst_id = vertexId;
	vs_out.depth = abs(vertexPosMV.z);
}