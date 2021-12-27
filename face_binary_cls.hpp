#pragma once

extern float conv0_weight[16*3*3*3];
extern float conv0_bias[16];

extern float conv1_weight[32*16*3*3];
extern float conv1_bias[32];

extern float conv2_weight[32*32*3*3];
extern float conv2_bias[32];

extern float fc0_weight[2*2048];
extern float fc0_bias[2];