#version 330 core

in vec4 vcolor;
flat in int frag_inst_id;

layout(location = 0) out vec4 color;
layout(location = 1) out int inst_id;

void main() {
	color = vcolor;
	inst_id = frag_inst_id;
}