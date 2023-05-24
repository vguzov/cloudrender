#version 330 core

in vec4 vcolor;

layout(location = 0) out vec4 color;
//layout(location = 1) out float inst_flag;

void main() {
//	color = vcolor;
//	inst_flag = 1.;
	color = vec4(vcolor.rgb,1.);
}