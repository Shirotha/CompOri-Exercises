double hessfuncxy(double A1, double A2, double A3, double A4, double A5){
    double R0_17 = 1.5;
    int I0_6 = 8;
    int I0_1 = -4;
    int I0_5 = 6;
    int I0_2 = -2;
    double R0_20 = 2.5;
    int I0_7 = 2;
    int I0_0 = -1;
    int I0_3 = 1;
    int I0_4 = 3;
    double R0_10 = 3.5;

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
    double R0_11;
    double R0_12;
    double R0_13;
    double R0_14;
    double R0_15;
    double R0_16;
    double R0_18;
    double R0_19;
    double R0_21;
    double R0_22;
    double R0_23;
    double R0_24;
    double R0_25;
    double R0_26;
    double R0_27;
    double R0_28;
    double R0_29;
    double R0_30;
    double R0_31;
    double R0_32;
    double R0_33;
    
    R0_0 = A1;
    R0_1 = A2;
    R0_2 = A3;
    R0_3 = A4;
    R0_4 = A5;
    R0_5 = (double) I0_0;
    R0_5 = R0_5 + R0_3;
    R0_6 = R0_4 * R0_4;
    R0_7 = R0_3 * R0_3;
    R0_8 = R0_7 + R0_6;
    R0_9 = (double) I0_1;
    R0_9 = R0_9 * R0_7;
    R0_11 = pow(R0_8, R0_10);
    R0_12 = 1 / R0_11;
    R0_11 = (double) I0_2;
    R0_11 = R0_11 * R0_3;
    R0_13 = (double) I0_1;
    R0_13 = R0_13 * R0_6;
    R0_14 = (double) I0_3;
    R0_14 = R0_14 + R0_11 + R0_7 + R0_6;
    R0_15 = pow(R0_14, R0_10);
    R0_16 = 1 / R0_15;
    R0_15 = pow(R0_8, R0_17);
    R0_18 = 1 / R0_15;
    R0_15 = R0_5 * R0_5;
    R0_19 = R0_15 + R0_6;
    R0_21 = pow(R0_8, R0_20);
    R0_22 = 1 / R0_21;
    R0_21 = pow(R0_19, R0_20);
    R0_23 = 1 / R0_21;
    R0_21 = (double) I0_4;
    R0_21 = R0_21 * R0_1 * R0_2 * R0_5 * R0_4 * R0_23;
    R0_24 = (double) I0_4;
    R0_24 = R0_24 * R0_0 * R0_2 * R0_3 * R0_4 * R0_22;
    R0_25 = R0_21 + R0_24;
    R0_26 = pow(R0_14, R0_20);
    R0_27 = 1 / R0_26;
    R0_26 = -R0_18;
    R0_28 = R0_0 * R0_2 * R0_3;
    R0_29 = pow(R0_19, R0_17);
    R0_30 = 1 / R0_29;
    R0_29 = R0_1 * R0_2 * R0_5 * R0_30;
    R0_30 = -R0_29;
    R0_29 = R0_0 * R0_2 * R0_3 * R0_18;
    R0_31 = -R0_29;
    R0_28 = R0_28 + R0_30 + R0_31;
    R0_30 = R0_9 + R0_6;
    R0_31 = R0_0 * R0_30 * R0_12;
    R0_30 = (double) I0_6;
    R0_30 = R0_30 * R0_3;
    R0_29 = (double) I0_1;
    R0_29 = R0_29 + R0_30 + R0_9 + R0_6;
    R0_30 = R0_1 * R0_29 * R0_16;
    R0_31 = R0_31 + R0_30;
    R0_30 = (double) I0_5;
    R0_30 = R0_30 * R0_2 * R0_4 * R0_28 * R0_31;
    R0_28 = R0_2 * R0_2;
    R0_31 = R0_7 + R0_13;
    R0_29 = R0_0 * R0_3 * R0_31 * R0_12;
    R0_31 = (double) I0_3;
    R0_31 = R0_31 + R0_11 + R0_7 + R0_13;
    R0_32 = R0_1 * R0_5 * R0_31 * R0_16;
    R0_29 = R0_29 + R0_32;
    R0_32 = R0_0 * R0_18;
    R0_31 = -R0_32;
    R0_32 = pow(R0_14, R0_17);
    R0_33 = 1 / R0_32;
    R0_32 = R0_1 * R0_33;
    R0_33 = -R0_32;
    R0_32 = R0_0 + R0_31 + R0_33;
    R0_31 = (double) I0_5;
    R0_31 = R0_31 * R0_28 * R0_4 * R0_29 * R0_32;
    R0_28 = (double) I0_1;
    R0_28 = R0_28 * R0_3;
    R0_29 = (double) I0_7;
    R0_29 = R0_29 * R0_7;
    R0_32 = -R0_6;
    R0_33 = (double) I0_7;
    R0_33 = R0_33 + R0_28 + R0_29 + R0_32;
    R0_28 = R0_1 * R0_2 * R0_33 * R0_27;
    R0_33 = (double) I0_4;
    R0_33 = R0_33 * R0_7 * R0_22;
    R0_29 = (double) I0_3;
    R0_29 = R0_29 + R0_33 + R0_26;
    R0_33 = R0_0 * R0_2 * R0_29;
    R0_28 = R0_28 + R0_33;
    R0_33 = (double) I0_7;
    R0_33 = R0_33 * R0_25 * R0_28;
    R0_28 = (double) I0_2;
    R0_28 = R0_28 * R0_6;
    R0_29 = (double) I0_3;
    R0_29 = R0_29 + R0_11 + R0_7 + R0_28;
    R0_28 = R0_1 * R0_2 * R0_29 * R0_27;
    R0_29 = -R0_28;
    R0_28 = (double) I0_4;
    R0_28 = R0_28 * R0_6 * R0_22;
    R0_32 = (double) I0_3;
    R0_32 = R0_32 + R0_28 + R0_26;
    R0_28 = R0_0 * R0_2 * R0_32;
    R0_29 = R0_29 + R0_28;
    R0_28 = (double) I0_7;
    R0_28 = R0_28 * R0_25 * R0_29;
    R0_30 = R0_30 + R0_31 + R0_33 + R0_28;

    return R0_30;
}

