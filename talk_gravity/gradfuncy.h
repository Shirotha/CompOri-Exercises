double gradfuncy(double A1, double A2, double A3, double A4, double A5){
    double R0_11 = 1.5;
    int I0_1 = -2;
    double R0_15 = 2.5;
    int I0_3 = 2;
    int I0_0 = -1;
    int I0_2 = 1;
    int I0_4 = 3;
    
    double R0_0;
    double R0_1;
    double R0_2;
    double R0_3;
    double R0_4;
    double R0_5;
    double R0_6;
    double R0_7;
    double R0_8;
    double R0_9;
    double R0_10;
    double R0_12;
    double R0_13;
    double R0_14;
    double R0_16;
    double R0_17;
    double R0_18;
    double R0_19;
    double R0_20;
    double R0_21;
    double R0_22;
    R0_0 = A1;
    R0_1 = A2;
    R0_2 = A3;
    R0_3 = A4;
    R0_4 = A5;
    R0_5 = (double) I0_0;
    R0_5 = R0_5 + R0_3;
    R0_6 = R0_4 * R0_4;
    R0_7 = R0_5 * R0_5;
    R0_8 = R0_7 + R0_6;
    R0_9 = R0_3 * R0_3;
    R0_10 = R0_9 + R0_6;
    R0_12 = pow(R0_10, R0_11);
    R0_13 = 1 / R0_12;
    R0_12 = (double) I0_1;
    R0_12 = R0_12 * R0_3;
    R0_14 = (double) I0_2;
    R0_14 = R0_14 + R0_12 + R0_9 + R0_6;
    R0_16 = pow(R0_10, R0_15);
    R0_17 = 1 / R0_16;
    R0_16 = pow(R0_8, R0_15);
    R0_18 = 1 / R0_16;
    R0_16 = (double) I0_4;
    R0_16 = R0_16 * R0_1 * R0_2 * R0_5 * R0_4 * R0_18;
    R0_18 = (double) I0_4;
    R0_18 = R0_18 * R0_0 * R0_2 * R0_3 * R0_4 * R0_17;
    R0_16 = R0_16 + R0_18;
    R0_18 = R0_0 * R0_2 * R0_3;
    R0_19 = pow(R0_8, R0_11);
    R0_20 = 1 / R0_19;
    R0_19 = R0_1 * R0_2 * R0_5 * R0_20;
    R0_20 = -R0_19;
    R0_19 = R0_0 * R0_2 * R0_3 * R0_13;
    R0_21 = -R0_19;
    R0_18 = R0_18 + R0_20 + R0_21;
    R0_20 = (double) I0_3;
    R0_20 = R0_20 * R0_16 * R0_18;
    R0_16 = R0_0 * R0_13;
    R0_18 = -R0_16;
    R0_16 = pow(R0_14, R0_11);
    R0_21 = 1 / R0_16;
    R0_16 = R0_1 * R0_21;
    R0_21 = -R0_16;
    R0_16 = R0_0 + R0_18 + R0_21;
    R0_18 = (double) I0_1;
    R0_18 = R0_18 * R0_6;
    R0_21 = (double) I0_2;
    R0_21 = R0_21 + R0_12 + R0_9 + R0_18;
    R0_18 = pow(R0_14, R0_15);
    R0_19 = 1 / R0_18;
    R0_18 = R0_1 * R0_2 * R0_21 * R0_19;
    R0_21 = -R0_18;
    R0_18 = (double) I0_4;
    R0_18 = R0_18 * R0_6 * R0_17;
    R0_19 = -R0_13;
    R0_22 = (double) I0_2;
    R0_22 = R0_22 + R0_18 + R0_19;
    R0_18 = R0_0 * R0_2 * R0_22;
    R0_21 = R0_21 + R0_18;
    R0_18 = (double) I0_3;
    R0_18 = R0_18 * R0_2 * R0_4 * R0_16 * R0_21;
    R0_20 = R0_20 + R0_18;

    return R0_20;
}
