#version 330 core

layout(location = 0) in vec2 pixelCoord;

uniform vec2 resolution;

void main(){
	gl_Position = vec4(2*pixelCoord/resolution-1, 0.5, 1);
}