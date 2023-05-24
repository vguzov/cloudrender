#version 330 core

layout(location = 0) out vec4 color;
uniform float depth_offset;

void main() {
//	color = vec4(gl_FragDepth,gl_FragDepth,gl_FragDepth,1);
	gl_FragDepth = gl_FragCoord.z + depth_offset;
	color = vec4(1,0,0,1);
}