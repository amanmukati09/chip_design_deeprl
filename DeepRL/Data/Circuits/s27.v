module s27(clk, G0, G1, G17, G2, G3);
  input clk, G0, G1, G2, G3;
  output G17;
  wire clk, G0, G1, G2, G3;
  wire G17;
  wire G5, G6, G7, n_0, n_1, n_2, n_3, n_4;
  wire n_5, n_6, n_7, n_8, n_9, n_10, n_11, n_12;
  wire n_20, n_21;
  fflopd DFF_0_Q_reg(.CK (clk), .D (n_12), .Q (G5));
  fflopd DFF_1_Q_reg(.CK (clk), .D (n_21), .Q (G6));
  not g543 (G17, n_20);
  not g545 (n_12, n_11);
  nand g546__7837 (n_11, G0, n_9);
  nor g548__7557 (n_10, n_7, n_8);
  nand g549__7654 (n_9, n_1, n_8);
  fflopd DFF_2_Q_reg(.CK (clk), .D (n_6), .Q (G7));
  nor g551__8867 (n_8, G7, n_4);
  not g553 (n_7, n_5);
  nor g550__1377 (n_6, G2, n_3);
  nand g554__3717 (n_5, n_2, G6);
  nand g552__4599 (n_4, G3, n_0);
  nor g555__3779 (n_3, G1, G7);
  not g557 (n_2, G0);
  not g556 (n_1, G5);
  not g558 (n_0, G1);
  nor g562__2007 (n_20, G5, n_10);
  nor g544_dup__1237 (n_21, G5, n_10);
endmodule



module fflopd(CK, D, Q);
  input CK, D;
  output Q;
  wire CK, D;
  wire Q;
  wire next_state;
  reg  qi;
  assign #1 Q = qi;
  assign next_state = D;
  always 
    @(posedge CK) 
      qi <= next_state;
  initial 
    qi <= 1'b0;
endmodule
