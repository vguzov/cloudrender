#version 330 core

in vec4 vcolor;
flat in int frag_inst_id;

layout(location = 0) out vec4 color;

void main(){
	color = vcolor;
}