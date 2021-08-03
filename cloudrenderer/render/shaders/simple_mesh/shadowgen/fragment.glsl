#version 330 core

in VS_OUT {
    float depth;
} fs_in;

layout(location = 0) out vec4 color;

void main() {
	color = vec4(1.);
}